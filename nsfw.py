import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
import time
import json
from typing import Dict, List, Optional, Union
import cv2
from deepface import DeepFace
import pickle
from urllib.parse import parse_qsl
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from urllib.parse import urlparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_device():
    """Determine device to use for computations"""
    if tf.config.list_physical_devices('GPU'):
        logging.info("Using GPU for computations")
        return "/GPU:0"
    logging.info("No GPU available, using CPU for computations")
    return "/CPU:0"

class LRUCache:
    def __init__(self, options: dict = None):
        if options is None:
            options = {}
        self.max = options.get('max', 100)
        self.maxAge = options.get('maxAge', 3600 * 1000)  # 1 hour default in milliseconds
        self.cache = {}
        self.timestamps = {}

    def get(self, key: str) -> Optional[dict]:
        if key not in self.cache:
            return None

        item = self.cache[key]
        timestamp = self.timestamps[key]

        if time.time() * 1000 - timestamp > self.maxAge:
            del self.cache[key]
            del self.timestamps[key]
            return None

        # Refresh item's timestamp
        self.timestamps[key] = time.time() * 1000
        return item

    def set(self, key: str, value: dict):
        if len(self.cache) >= self.max:
            oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]

        if 'results' in value:
            value['results'] = [{
                **r,
                'probability': round(r['probability'] * 100) / 100
            } for r in value['results']]

        self.cache[key] = value
        self.timestamps[key] = time.time() * 1000

class NsfwClassifier:
    def __init__(self):
        self.MODEL_PATH = './models/model.h5'
        self.classes = ['drawing', 'hentai', 'neutral', 'porn', 'sexy']
        self.device = get_device()
        # Load model in initialization
        with tf.device(self.device):
            try:
                self.model = tf.keras.models.load_model(self.MODEL_PATH)
                logging.info("NSFW image classifier model loaded.")
            except Exception as e:
                logging.error(f"Error loading NSFW model: {e}")
                raise

    def classify_image(self, image: Image.Image):
        try:
            # Resize and preprocess image
            image = image.resize((224, 224))
            img_array = np.array(image).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Get predictions using the specified device
            with tf.device(self.device):
                predictions = self.model.predict(img_array)

            # Process results
            results = [
                {
                    'className': class_name,
                    'probability': float(pred)
                }
                for class_name, pred in zip(self.classes, predictions[0])
            ]

            # Sort results by probability
            results.sort(key=lambda x: x['probability'], reverse=True)

            # Determine if NSFW
            is_nsfw = (
                predictions[0][1] > 0.2 or  # hentai
                predictions[0][3] > 0.2 or  # porn
                predictions[0][4] > 0.2     # sexy
            )

            return {
                'isNSFW': is_nsfw,
                'results': results
            }

        except Exception as e:
            logging.error(f"Error during classification: {e}")
            raise

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class NsfwTextClassifier:
    def __init__(self):
        self.MODEL_PATH = './models/nsfw_classifier.h5'
        self.TOKENIZER_PATH = './models/nsfw_classifier_tokenizer.pickle'
        self.maxSequenceLength = 50
        self.device = get_device()

        # Load model and tokenizer in initialization
        try:
            with tf.device(self.device):
                # Load model and compile
                self.model = tf.keras.models.load_model(
                    self.MODEL_PATH,
                    compile=False
                )
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

            # Load tokenizer from pickle file
            with open(self.TOKENIZER_PATH, 'rb') as f:
                self.tokenizer = pickle.load(f)
            logging.info("NSFW text classifier model and tokenizer loaded.")
        except Exception as e:
            logging.error(f"Error loading text classifier model or tokenizer: {e}")
            raise

    def preprocess(self, text, isfirst=True):
        if isfirst:
            if isinstance(text, str):
                pass
            elif isinstance(text, list):
                output = []
                for i in text:
                    output.append(self.preprocess(i))
                return output

        text = re.sub('<.*?>', '', text)
        text = re.sub('\(+', '(', text)
        text = re.sub('\)+', ')', text)
        matches = re.findall('\(.*?\)', text)

        for match in matches:
            text = text.replace(match, self.preprocess(match[1:-1], isfirst=False))

        text = text.replace('\n', ',').replace('|', ',')

        if isfirst:
            output = text.split(',')
            output = [x.strip() for x in output if x.strip() != '']
            return ', '.join(output)

        return text

    def tokenize_text(self, text: str) -> List[int]:
        preprocessed_text = self.preprocess(text)
        sequence = self.tokenizer.texts_to_sequences([preprocessed_text])
        padded = pad_sequences(sequence, maxlen=self.maxSequenceLength, padding='post', truncating='post')
        return padded[0]

    def classify_text(self, prompt: str, negative_prompt: str = '') -> Dict[str, Union[bool, float]]:
        prompt_tokens = self.tokenize_text(prompt)
        neg_prompt_tokens = self.tokenize_text(negative_prompt)

        prompt_tensor = np.array([prompt_tokens])
        neg_prompt_tensor = np.array([neg_prompt_tokens])

        # Use the specified device for prediction
        with tf.device(self.device):
            prediction = self.model.predict([prompt_tensor, neg_prompt_tensor])

        score = float(prediction[0][0])

        return {
            'isNSFW': score > 0.5,
            'score': score
        }

class FaceDetector:
    def __init__(self):
        self.device = get_device()
        logging.info("DeepFace initialized for face detection.")

    def detect_faces(self, image: Image.Image) -> list:
        # Convert PIL image to numpy array in BGR format for OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = []

        try:
            # Use DeepFace.analyze with only age detection
            analysis = DeepFace.analyze(
                img_path=image_cv,
                actions=['age'],  # Only age detection
                enforce_detection=False,
                detector_backend='opencv'
            )

            # DeepFace.analyze returns a list of dictionaries
            if isinstance(analysis, list):
                results = analysis
            else:
                results = [analysis]

            for result in results:
                faces.append({
                    "age": result.get('age')
                })
                # Print only age
                print(f"Age detected: {result.get('age')} years")

        except Exception as e:
            logging.error(f"Error during face detection: {e}")

        return faces

class NsfwDetectorClass:
    def __init__(self):
        self.nsfw_classifier = NsfwClassifier()
        self.text_classifier = NsfwTextClassifier()
        self.face_detector = FaceDetector()
        self.cache = LRUCache({'max': 500, 'maxAge': 300000})  # 5 minutes cache

        # Image processing settings
        self.image_settings = {
            'maxSize': 512,
            'progressiveSizes': [256, 512],
            'confidenceThresholds': {
                256: 0.8,
                512: 0.6
            }
        }

        # Load keywords
        self.nsfw_keywords = self.load_nsfw_keywords()
        self.under20_keywords = self.load_under20_keywords()

        # Compile regex patterns
        self.nsfw_keyword_regex = re.compile(r'\b(' + '|'.join(map(re.escape, self.nsfw_keywords)) + r')\b', re.IGNORECASE)
        self.under20_keyword_regex = re.compile(r'\b(' + '|'.join(map(re.escape, self.under20_keywords)) + r')\b', re.IGNORECASE)

        # Create output directories if they don't exist
        self.output_dirs = {
            'sfw': Path('./processed_images/sfw'),
            'nsfw': Path('./processed_images/nsfw')
        }
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        self.processed_links = {
            'success': [],
            'failed': [],
            'skipped': [],
            'keyword_blocked': [],
            'text_blocked': []
        }

    def contains_keywords(self, text: str) -> bool:
        if not text:
            return False
        lower_text = text.lower()
        return bool(self.nsfw_keyword_regex.search(lower_text) or self.under20_keyword_regex.search(lower_text))

    def load_nsfw_keywords(self) -> List[str]:
        # Full list of NSFW keywords
        return [
            # NSFW keywords list
            "hardcore","kiss","kissing", "obscene", "nude", "nudity", "naked", "sensual", "provocative", 
            "suggestive", "fetish", "kink", "voyeur", "seductive", "sexual", "lustful", 
            "sultry", "risque", "taboo", "voyeuristic", "undressed", "unclothed", "bare", 
            "exposed", "intimate", "raunchy", "strip", "stripping", "stripped", "undressing", 
            "breasts", "boobs", "tits", "nipples", "genitals", "vagina", "penis", "testicles", 
            "scrotum", "pubic", "buttocks", "ass", "groin", "crotch", "thighs", "hips", 
            "threesome", "orgy", "orgasm", "intercourse", "masturbation", "foreplay", 
            "oral", "blowjob", "ejaculation", "penetration", "incest", "molest", 
            "molestation", "rape", "raping", "pedo", "pedophile", "child abuse", 
            "child pornography", "underage", "loli", "cp", "young girl", "young boy", 
            "schoolgirl", "schoolboy", "teen", "minor", "jailbait", "child", "infant", 
            "baby", "toddler", "pubescent", "prepubescent", "underaged", "innocent", 
            "ch1ld", "ch!ld", "fet1sh", "masturb@t1on", "int1m@te", "h@rdcore", 
            "rap3", "mol3st", "undr3ssed", "int1m@te", "ej@cul@tion", "p3n!s", "gen!tal", 
            "bre@st", "b00bs", "n!pples", "h1ps", "b0dy", "v@gina", "sc@ntily", 
            "sl@ve", "dom!nant", "b0dy", "t!ts", "c0rs3t", "br@", "p@nty", "br@less", 
            "exposed", "corset", "lingerie", "thong", "panties", "sheer", "see-through", 
            "topless", "braless", "stripper", "strippers", "lap dance", "camgirl", "camgirls", 
            "dominatrix", "submissive", "degrading", "degrade", "abuse","abuser", "abusive", "shame", "shaming", 
            "bondage", "bdsm", "slave", "master", "mistress", "discipline", "spank", 
            "whip", "gag", "choke", "handcuff", "restraint", "leather", "latex", 
            "g-string", "garter", "fishnet", "stockings", "thigh-highs", "corset", 
            "dominant", "submissive", "roleplay", "sex", "hardcore sex", "softcore", 
            "soft porn", "hard porn", "cam", "webcam", "onlyfans", "fansly", "sex work", 
            "prostitute", "escort", "escort services", "hooker", "hooking", "massage parlor", 
            "happy ending", "escort services", "body rubs", "massage", "nuru", "erotica", 
            "bdsm", "domination", "submissive", "sadomasochism", "sadism", "masochism", 
            "gore", "snuff", "beheading", "brutality", "torture", "gruesome", "disgusting", 
            "grotesque", "bloody", "mutilation", "self-harm", "suicide", "killing", 
            "murder", "slaughter", "injure", "injury", "rape scene", "porn video", 
            "sex tape", "sex video", "creampie", "facial", "gagging","bath","shower", "rough sex", 
            "anal", "fisting", "gaping", "penetrate", "penile", "vaginal", "g-string", 
            "seduce", "seducing", "intercourse", "copulation", "intimacy", "intimate", 
            "intimately", "making out", "make out", "foreplay", "tease", "aroused", 
            "arousal", "horny", "desire", "kama sutra", "doggy style", "missionary", 
            "cowgirl", "reverse cowgirl", "69 position", "dominant position", "top position", 
            "bottom position", "sexual fantasy", "fantasy", "fetishist", "fetishistic", 
            "foot fetish", "lingerie model", "intimate apparel", "racy", "sizzling", 
            "tempting", "inviting", "pornstar", "centerfold", "hustler", "playboy", 
            "playmate", "penthouse", "eroticism", "suggestive", "naughty", "sultry", 
            "lustful", "flirty", "flirting", "provoking", "provocative", "tempting", 
            "temptress", "seduction", "vixen", "temptress", "desirable", "sexiness", 
            "alluring", "lust", "dirty talk", "sexy talk", "sexting", "dirty message", 
            "x-rated", "triple-x", "slut", "whore", "tramp", "harlot", "bimbo", 
            "swinger", "swinging", "swingers", "seduction", "flirt", "flirting", 
            "seductress", "lusting", "hot", "hottie", "sexual encounter", "hookup", 
            "sizzling hot", "heated", "arouse", "aroused", "horny", "seductive", 
            "come-hither", "racy", "rave", "body shots", "strip show", "peep show", 
            "sexual innuendo", "innuendo", "sexual overtones", "hot chick", "hot guy", 
            "hot girl", "playgirl", "adult movie", "adult site", "erotic movie", 
            "adult content", "bareback", "buttplug", "anal beads", "dildo", "vibrator", 
            "sex toy", "strap-on", "sex slave", "dominatrix", "latex","bra","bondage gear",
            "lingerie", "swimwear", "bikini","bathtub","sweaty",
            "fuck", "shit", "cock", "tidd", "clit", "dick", "pussy", "boob", "testicl", 
            "nigga", "chink", "jew", "brothel", "undies", "birthday suit", "gang", 
            "cumshot", "cum", "niple", "lana", "jayden", "riley reid", "breed", "dani", 
            "nip slip", "nipslip", "cunt", "dong", "magnum", "snort", "crack", "cocaine", 
            "nymph", "nigger", "hunk", "lewd", "fellatio", "vulva", "pedophi", "paedophi", 
            "milf", "cunnilingus", "exspose", "upskirt", "pawg", "bbw", "oppai", "pechos", 
            "desnuda", "ava addams", "angela white", "milkers", "succubus", "tiddies", 
            "coitus", "shagging", "b00b", "n0de", "gluteus maximus", "brazzer", "ahegao", 
            "petajansen", "hentia", "botty", "titten", "ecchi doujin", "alison tyler", 
            "alura jenson", "chesticle", "sloot", "cadaver", "bewbs", "fornicat", "phallus", 
            "twat", "squabitis", "bushless", "twinks", "ballbust", "squirting", "hotwife",
            "bangbro", "futanari", "cameltoe", "beastiality", "giantess", "butthole", 
            "facesit", "castrated", "hogtie", "vigina", "henati", "diaper", "urinat",
            "maggot", "vaggina", "licck", "ppenis", "cguming", "femdom", "k18", "rectal"            # ... (continue with the full list)
        ]
    def load_under20_keywords(self) -> List[str]:
        # Full list of under-20 keywords
        return [
            "adolescent", "baby", "birthday party", "boy", "child", "classmate", "daycare", 
            "freshman", "girl", "high school", "infant", "junior", "kid", "kids", "kindergartener", 
            "little boy", "little girl", "middle school", "minor", "preschool", "primary school", 
            "pubescent", "schoolboy", "schoolgirl", "second grader", "senior", "sophomore", 
            "student", "teen", "teenager", "toddler", "under 20", "underage", "young adult", 
            "youngster", "youth",

            #Additional words from underageKeywords
            "tween", "preteen", "preadolescent", "juvenile", "pupil", "preschooler", "elementary",

            #Additional words from gradeKeywords
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "eleventh", "twelfth", "1st", "2nd",
            "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",

            #Additional words from ageKeywords
            "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen","cutie",
            "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"            # ... (continue with the full list)
        ]



    def convert_hotpot_link_to_s3(self, hotpot_link: str) -> Optional[str]:
        match = re.search(r'/art-generator/(8-[\w\d]+)', hotpot_link)
        if match:
            return f"https://hotpotmedia.s3.us-east-2.amazonaws.com/{match.group(1)}.png"
        logging.warning(f"Could not convert link to S3 URL: {hotpot_link}")
        return None

    def analyze_image(self, image_url: str) -> Dict[str, Union[bool, float, int]]:
        try:
            result = self.process_image_progressive(image_url)
            # Convert NumPy types to Python native types
            return {
                'isNSFW': bool(result['isNSFW']),
                'confidence': float(result.get('confidence', 0)),
                'resolution': int(result.get('resolution', 0))
            }
        except Exception as e:
            logging.error(f"Error analyzing image: {e}")
            raise

    def process_image_progressive(self, image_url: str) -> Dict[str, Union[bool, float, int]]:
        progressive_sizes = self.image_settings['progressiveSizes']
        confidence_thresholds = self.image_settings['confidenceThresholds']

        result = None

        try:
            image = self.load_image_from_url(image_url)

            for size in progressive_sizes:
                # Resize image
                width, height = self.calculate_dimensions(image, size)
                resized_img = image.resize((width, height))

                # Classify
                current_result = self.nsfw_classifier.classify_image(resized_img)
                confidence = max(r['probability'] for r in current_result['results'])

                # Early return if confident enough
                if confidence > confidence_thresholds[size]:
                    result = {
                        'isNSFW': current_result['isNSFW'],
                        'confidence': confidence,
                        'resolution': size
                    }
                    break

                result = {
                    'isNSFW': current_result['isNSFW'],
                    'confidence': confidence,
                    'resolution': size
                }

            return result

        except Exception as e:
            logging.error(f"Error processing image: {e}")
            raise e

    def load_image_from_url(self, image_url: str) -> Image.Image:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image

    @staticmethod
    def calculate_dimensions(img: Image.Image, max_size: int) -> tuple:
        ratio = min(max_size / img.width, max_size / img.height)
        return (
            round(img.width * ratio),
            round(img.height * ratio)
        )

    def save_image(self, image_url: str, is_nsfw: bool, original_link: str):
        try:
            # Download image
            response = requests.get(image_url)
            if response.status_code == 200:
                # Get filename from URL
                filename = os.path.basename(urlparse(image_url).path)
                
                # Determine target directory
                target_dir = self.output_dirs['nsfw'] if is_nsfw else self.output_dirs['sfw']
                
                # Save image
                image_path = target_dir / filename
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                self.processed_links['success'].append({
                    'original_link': original_link,
                    'image_url': image_url,
                    'saved_path': str(image_path),
                    'is_nsfw': is_nsfw
                })
                print(f"Image saved to: {image_path}")
            else:
                self.processed_links['failed'].append({
                    'original_link': original_link,
                    'image_url': image_url,
                    'reason': f"Download failed with status {response.status_code}"
                })
                print(f"Failed to download image: {image_url}")
        except Exception as e:
            self.processed_links['failed'].append({
                'original_link': original_link,
                'image_url': image_url,
                'reason': str(e)
            })
            print(f"Error saving image: {e}")

    def is_nsfw(self, hotpot_link: str) -> Dict[str, Union[bool, str, float, list]]:
        # Check cache first
        cached_result = self.cache.get(hotpot_link)
        if cached_result:
            return cached_result

        # Parse URL and get title
        url = requests.utils.urlparse(hotpot_link)
        params = dict(parse_qsl(url.query))
        title = params.get('title', '')

        # Get S3 URL first (we'll need it for all cases)
        image_url = self.convert_hotpot_link_to_s3(hotpot_link)
        if not image_url:
            self.processed_links['failed'].append({
                'link': hotpot_link,
                'reason': 'Failed to convert to S3 URL'
            })
            return {'isNSFW': True, 'reason': 'Processing error'}

        # Step 1: Check for NSFW keywords
        if self.nsfw_keyword_regex.search(title.lower()):
            result = {
                'isNSFW': True, 
                'reason': 'Keyword match',
                'imageUrl': image_url  # Add image URL
            }
            self.processed_links['keyword_blocked'].append(hotpot_link)
            self.save_image(image_url, True, hotpot_link)  # Save as NSFW
            return result

        # Step 2: Text Classification
        text_result = self.text_classifier.classify_text(title)
        if text_result['isNSFW']:
            result = {
                'isNSFW': True, 
                'reason': 'Text classification',
                'imageUrl': image_url  # Add image URL
            }
            self.processed_links['text_blocked'].append(hotpot_link)
            self.save_image(image_url, True, hotpot_link)  # Save as NSFW
            return result

        # Step 3: Image Classification
        try:
            image_analysis = self.analyze_image(image_url)
            result = {
                'isNSFW': bool(image_analysis['isNSFW']),  # Convert np.bool_ to Python bool
                'reason': 'Image classification',
                'confidence': float(image_analysis.get('confidence', 0)),  # Convert to Python float
                'imageUrl': image_url
            }
            self.save_image(image_url, bool(image_analysis['isNSFW']), hotpot_link)  # Convert bool_ here too
            return result
        except Exception as e:
            self.processed_links['failed'].append({
                'link': hotpot_link,
                'reason': f'Image analysis error: {str(e)}'
            })
            return {'isNSFW': True, 'reason': 'Processing error'}

def main():
    detector = NsfwDetectorClass()
    
    print(f"Images will be saved to:")
    print(f"SFW: {detector.output_dirs['sfw']}")
    print(f"NSFW: {detector.output_dirs['nsfw']}")
    
    # Test links
    test_links = [
        # Add your list of links here
        "https://hotpot.ai/s/art-generator/8-PO0PLZjhDCwypig?share=1&title=flood",
        "https://hotpot.ai/s/art-generator/8-eoKHdJ5dLMkyp8b?share=1&title=Purple%20male%20angel%20with%20four%20wings",
        "https://hotpot.ai/s/art-generator/8-PvtmI8H4ZuaCHbz?share=1&title=40yo%20german%20woman.%20flabby%20body",
        "https://hotpot.ai/s/art-generator/8-l7GpYVYJERZ7yeU?share=1&title=Beautiful%20fat%20blonde%20Japanese%20barbarian%20woman%20reclining%20in%20a%20bath",
        "https://hotpot.ai/s/art-generator/8-64tcsutdceJnO00?share=1&title=40yo%20german%20woman.%20fatless%20body",
        "https://hotpot.ai/s/art-generator/8-gsOZUc2SqmIt688?share=1&title=Skinny%2C%20anorexic%20and%20young%20girl%20in%20white%2C%20very%20thin%2C%20look%20through%2C%20too%20small%2C%20in%20skintone%20leotard%20on%20knees%20during%20gymnastic%20split%20exercise.%20Camera%20is%20on%20floor%2C%20view%20at%20front%20(zoomed%20in)%20of%20girl.%20Leotard%20is%20fully%20transparent%2C%20crystalline%20and%20lucent.",
        "https://hotpot.ai/s/art-generator/8-Nu9YFE56W5LeBZQ?share=1&title=deadpool",
        "https://hotpot.ai/s/art-generator/8-SMzwdybyECoshgu?share=1&title=Cutie%20female%20on%20a%20pink%20princess%20play%20mat%20Gothic%20showing%20off%20pelvic%20mound",
        "https://hotpot.ai/s/art-generator/8-SYDmpTTtbW8p7cN?share=1&title=Hotprincessjasmine%20with%20glasses%20and%20with%20black%20and%20straight%20hair%2C%20hot%20torso%2C%20pale%20and%20white%20skin%2C%20hot%20cosplay%20orgi",
        "https://hotpot.ai/s/art-generator/8-VEBwN3q11ixwJdY?share=1&title=Cutie%20female%20on%20a%20pink%20princess%20play%20mat%20Gothic%20showing%20off%20pelvic%20mound",
        "https://hotpot.ai/s/art-generator/8-XzamKqvHPiON5Zl?share=1&title=A%20blonde%20woman%20with%20big%20chest%20height%20comparison",
        "https://hotpot.ai/s/art-generator/8-rv1KLwsiQGtvcQB?share=1&title=turks%20cars",
        "https://hotpot.ai/s/art-generator/8-sDFgCxfSqDmrsnc?share=1&title=Rogue%20assassin%20malard",
        "https://hotpot.ai/s/art-generator/8-ROTQMwDK5Zrs3MI?share=1&title=half%20elf%20bard%20female%20taylor%20swift%20headshot%20lute",
        "https://hotpot.ai/s/art-generator/8-N7gK7fM7CVFrqd9?share=1&title=a%20bathtub%20viewed%20from%20above%20a%20person%20inside%20red%20lipped",
        "https://hotpot.ai/s/art-generator/8-k8hbQZVI8RBRnhs?share=1&title=Dr%20Seuss%20cat%20in%20the%20hat%20tabaxi",
        "https://hotpot.ai/s/art-generator/8-hHrxugX90apU2CK?share=1&title=a%20bathtub%20viewed%20from%20above%20a%20person%20inside",
        "https://hotpot.ai/s/art-generator/8-8xn4qFQZCt3U3i7?share=1&title=Bibi%20Nethanihu",
        "https://hotpot.ai/s/art-generator/8-Z0ogAOyNTQexQRn?share=1&title=reimagine%20a%20camping%20tent%20that%20is%20made%20coco%20channel%20logo%20and%20style%20it%27s%20black%20and%20torn%20with%20a%20golden%20logo%20on%20it%20Channel%20it%20is%20set%20up%20in%20the%20street%20for%20a%20beggar",
        "https://hotpot.ai/s/art-generator/8-4QO6FfzF19wqw3z?share=1&title=old%20westerner%20with%20banjo%20on%20fire%20in%20chain%20mail%20surrounded%20by%20goblins",
        "https://hotpot.ai/s/art-generator/8-SP7unigZL7Jo9Xn?share=1&title=Urumqi",
        "https://hotpot.ai/s/art-generator/8-qFP40wnR8dUAwUM?share=1&title=the%20equation%20of%20intelligent%20life",
        "https://hotpot.ai/s/art-generator/8-VpEZ8C6K9nvH1WK?share=1&title=Alle%20Kinder%20rannten%20%C3%BCber%20das%20Feld%2C%20bis%20auf%20ihn%20wurde%20er%20von%20einer%20Kuh%20Vergewaltigt",
        "https://hotpot.ai/s/art-generator/8-bQ2HWPpQDPoFgTB?share=1&title=A%20girl%20looking%20to%20the%20othherside%20of%20the%20river%20looking%20at%20golden%20fish",   
        "https://hotpot.ai/s/art-generator/8-nY5QO2gsLRAFs3b?share=1&title=the%20equation%20of%20intelligent%20life",
        "https://hotpot.ai/s/art-generator/8-MWEcDwxuXi1sXGn?share=1&title=young%20handsome%20British%20gay%20couple%2C%20hugging%2C%20beautiful%20shapely%20hips%2C%20extreme%20tight%20super%20skinny%20tan%20denim%2C%20tattoos%2C%20crew%20haircut",
        "https://hotpot.ai/s/art-generator/8-6MH1BtQweieYDTm?share=1&title=Gold%20giant%20sea%20dragon%20in%20the%20artic",
        "https://hotpot.ai/s/art-generator/8-8aYxxLr3hX1NzCd?share=1&title=people%20in%20a%20bathtub%20viewed%20from%20above%20Let%20the%20people%20inside%20be%20only%20female%20let%20her%20hair%20be%20blonde",
        "https://hotpot.ai/s/art-generator/8-4utZYpYpBVS99lA?share=1&title=shrek%20dancing%20on%20a%20moutain%20eating%20soup",
        "https://hotpot.ai/s/art-generator/8-mYpS3npzVdTdcP5?share=1&title=Woman%20on%20beach%20Ettingbo%20shawarma",
        "https://hotpot.ai/s/art-generator/8-7QgB5fVtoDuQN1I?share=1&title=banner%203d%20print%20service%20word",
        "https://hotpot.ai/s/art-generator/8-onHBvVtPSwRwExd?share=1&title=people%20in%20a%20bathtub%20viewed%20from%20above%20Let%20the%20people%20inside%20be%20only%20female%20let%20her%20hair%20be%20blonde",
        "https://hotpot.ai/s/art-generator/8-B7NIEtK8kmdqpyg?share=1&title=an%20enormous%20castle%20atop%20a%20cliff%20surrounded%20by%20forest",
        "https://hotpot.ai/s/art-generator/8-ROTQMwDK5Zrs3MI?share=1&title=half%20elf%20bard%20female%20taylor%20swift%20headshot%20lute",
        "https://hotpot.ai/s/art-generator/8-qhnXEdqpbbdupOB?share=1&title=muppet%20and%20baby%20muppet",
        "https://hotpot.ai/s/art-generator/8-VN6ENJhhmqOt8ap?share=1&title=black%20hellhounds",
        "https://hotpot.ai/s/art-generator/8-TSDq4g86ydoA9sf?share=1&title=bottom%20view%20of%20sweaty%20wet%20muscular%20tanned%20ponytail%20taylor%20swift%2C%20in%20tactical%20army%20shorts%2C%20hamstrings%2C%20headset%20reclining%20in%20external%20gun%20bubble%20inside%20plastic%20army%20glider%20in%20high%20orbit%2C%20bullet%20casings%20flying",
        "https://hotpot.ai/s/art-generator/8-YaMqKqLAHrYuHg3?share=1&title=bottom%20view%20of%20sweaty%20wet%20muscular%20tanned%20ponytail%20taylor%20swift%2C%20in%20tactical%20army%20shorts%2C%20hamstrings%2C%20headset%20reclining%20in%20external%20gun%20bubble%20inside%20plastic%20army%20glider%20in%20high%20orbit%2C%20bullet%20casings%20flying",
        "https://hotpot.ai/s/art-generator/8-EPJ8Ag6voJ6bUnJ?share=1&title=Solarion%20is%20an%20ethereal%20being%20with%20a%20swirling%20form%20in%20bright%20greens%2C%20yellows%2C%20and%20blues%2C%20adorned%20with%20shimmering%2C%20leaf-like%20patterns%20and%20glowing%20solar%20motifs.%20They%20have%20tendrils%20resembling%20glowing%20vines%20for%20reaching%20and%20absorbing%20light%2C%20while%20their%20core%20emits%20a%20vibrant%20energy%20glow.%20Luminous%20energy%20orbs%20hover%20around%20them%2C%20ready%20to%20be%20projected%2C%20enhancing%20their%20dynamic%20presence.%20In%20car%20like%20form",
        "https://hotpot.ai/s/art-generator/8-LPhPH9uu1VGU6sC?share=1&title=an%20assassin%20with%20a%20gun%20that%20shoots%20green%20farts",
        "https://hotpot.ai/s/art-generator/8-2bqQgfRutl1ibSO?share=1&title=N%40ked%20woman%2C%20full%20pic%20i.e%20front%20up%20to%20lower%20body",
        "https://hotpot.ai/s/art-generator/8-zLPIUsU8joGmSdC?share=1&title=vintage%20grainy%2080s%20desaturated%20upper%20body%20photo%2C%2080s%20medium%20hairstyle%2C%20chubby%20body%2C%20cute%20Japanese%20young%20woman%20dressed%20in%20cobalt%20blue%20skimp%20sleeveless%20v-neck%20high-cut%20cutout%20shaped%20monokini%20with%20red%20mantle%20cape%2C%20city%20background",
        "https://hotpot.ai/s/art-generator/8-SicOtnbH1Hz6DhM?share=1&title=beautiful%20curvy%20mature%20redhead%20warrior%20woman%2C%20flesh%20body%2C%20long%20hair%2C%20big%20belly%2C%20behind%20view%2C%20in%20courtyard",
        "https://hotpot.ai/s/art-generator/8-iH0sOuFLg0i3ftG?share=1&title=Woman%20sitting%20on%20beach%20holding%20a%20seashell",
        "https://hotpot.ai/s/art-generator/8-yUaITDjJw0XH5jL?share=1&title=samurai%20warrior%20armor%20robot%20high%20tech%20red%20armor",
        "https://hotpot.ai/s/art-generator/8-9vXDH7VH1Pnia5h?share=1&title=extreme%20closeup%20shot%20of%20two%20beautiful%20girlfriends%20dressed%20in%20purple%20bandeau%20crop%20tops%20hugging%20each%20other%20and%20smiling%20to%20the%20camera%20laying%20in%20bed",
        "https://hotpot.ai/s/art-generator/8-s7T7CIeE2CpiTTb?share=1&title=Three%20full-bodied%2C%20youthful%2C%20stunningly%20gorgeous%2C%20young%2C%20Anglican%20girls%20with%20flowing%20hair.%0A%0AThey%20are%20wearing%20a%20fleece%20quilted%20jersey%20pullover%20dress%20with%20thigh-high%20sheepskin%20Uggs%20boots.%0A%0AThey%20are%20smiling%2C%20standing%2C%20and%20facing%20toward%20the%20camera.%2C%20high-angle%20shot",
        "https://hotpot.ai/s/art-generator/8-YqFKzkb1m0nSuQL?share=1&title=canoe%20with%20forest%20silhouette%20in%20the%20background",
        "https://hotpot.ai/s/art-generator/8-5lLrtkB6JAJRu1r?share=1&title=vintage%2080s%20desaturated%20upper%20body%20photo%2C%2080s%20medium%20hairstyle%2C%20chubby%20body%2C%20cute%20Japanese%20young%20woman%20dressed%20in%20cobalt%20blue%20skimp%20sleeveless%20v-neck%20high-cut%20cutout%20shaped%20monokini%20with%20red%20mantle%20cape%2C%20green%20neon%20spotlight%20room%20background",
        "https://hotpot.ai/s/art-generator/8-BdExH7hywFwEZZi?share=1&title=Muscled%20Mike%20Shinoda%20at%2036%20wearing%20elegant%20black%20suit%20slightly%20smiling%20in%20rome",
        "https://hotpot.ai/s/art-generator/8-3SzuC2nYIPR11C4?share=1&title=Shocking%20Transformation%3A%20Blonde%20College%20Girl%27s%20Weight%20Gain%20fat%20belly",
        "https://hotpot.ai/s/art-generator/8-fdjrZI9XOi2Hgc9?share=1&title=Handsome%20British%20gay%20couple%2C%20hugging%2C%20shapely%20hips%2C%20extreme%20tight%20super%20skinny%20tan%20denim%2C%20tattoos%2C%20crew%20haircut",
        "https://hotpot.ai/s/art-generator/8-oKBrBvcxCdhpHfx?share=1&title=Beautiful%20black%20barbarian%20woman%20reclining%20in%20a%20bath",
        "https://hotpot.ai/s/art-generator/8-Rj2WedQOyAcQXD0?share=1&title=Not%20for%20Minors%7BIllegally%20Sinful%3A%20Near%20Future%2C%20S%26M%20Transparent-DarXxyneSs-Datex-Amethyst%2C%20Kunoichi-SorcereSs/Star%20Trek%20Vulcan%2C%20Mirror%20Universe%2C%20Tera%20Patrick%20Sensitively%20Sensitive%20Erogenousing%20Taimanin%20AsagI%20Lilith%20Soft%20Waifu-EnchantreSsor.%20Exotic%20Love%20Lovense%20FEMDOMming%7DK18%20HoTt%20Deluxe%20Kink%20Extra%7BNaught%7By%7D%7D%7BHenta%7Bi%7D%7D.%20Diamond-Secret-Love%7B%7BK18%7D%2B%7DDEePp%20Coffee-OiLled%20OiLly%20%7BSuccubu%7BSs%7D%7DSkin",
        "https://hotpot.ai/s/art-generator/8-0dG3pJUHVJevefH?share=1&title=Black%20Hoodie%20Black%20Pants%20Black%20long%20hair%20chain%20wallet",
        "https://hotpot.ai/s/art-generator/8-EdiEl6ClkLq7m9Z?share=1&title=Thick%20leg%20Wonder%20Woman%20rubbing%20moisturizer%20under%20skirt",
        "https://hotpot.ai/s/art-generator/8-ZqPkjvoKl77SBoz?share=1&title=human%20woman%27s%20body%20with%20a%20cat%27s%20tail%20and%20fur%20growing%20out%20of%20her%20backside",
        "https://hotpot.ai/s/art-generator/8-WrbCLg881a4F6fZ?share=1&title=Thick%20leg%20Wonder%20Woman%20rubbing%20moisturizer%20on%20her%20b%C3%B2obs",
        "https://hotpot.ai/s/art-generator/8-cnqfdKlJ0pBXTI8?share=1&title=extreme%20low-angle%20extreme%20close-up%20a%20strong%20dark%20powerful%20big%20plump%20angry%20italian%20woman%20behind%20bars%20short%20bobbed%20shoulder-length%20hair%20slightly%20large%20nose%20dark%20blue%20polyester%20minidress%20with%20white%20pattern%20leaning%20doubled%20over%20us%20arms%20out%20closing%20down%20iron%20cage%20doors%20on%20us%2C%20ceiling%20light%20behind%2C%20suburban%20living%20room",
        "https://hotpot.ai/s/art-generator/8-lGi55ShKCGGhozf?share=1&title=Metallica%20going%20rollerblading",
        "https://hotpot.ai/s/art-generator/8-eNa1OHHrejZuNOl?share=1&title=extremely%20low-angle%20strong%20dark%20powerful%20big%20plump%20irritated%20agnetha%20falkstog%20in%20her%2020s%20with%20whip%2C%20totally-straight%20hair%20high%20forehead%20puckered%20full%20lips%20dark%20brown%20eyes%20long%20eyelashes%2C%20doubled%20over%20arm%20outstretched%20beating%20us%20violently%2C%20black%20lace%20dress%2C%20against%20a%20bright%20greenscreen%20background",
        "https://hotpot.ai/s/art-generator/8-hnu19bF1FAiAB6J?share=1&title=Goblin%20arcane%20archer%20with%20green%20skin%2C%20red%20eyes%2C%20black%20hair%2C%20wearing%20a%20black%20leather%20jacket%20with%20a%20red%20scarf%20around%20her%20neck%2C%20holding%20a%20long%20sword%20with%20a%20red%20tassle%20on%20the%20end",
        "https://hotpot.ai/s/art-generator/8-GiqrblNsncJNND9?share=1&title=extreme%20low-angle%20extreme%20close-up%20of%20face%20of%20two%20strong%20dark%20powerful%20big%20plump%20sweaty%20italian%20women%20short%20bobbed%20shoulder-length%20hair%20slightly%20large%20nose%20full%20lips%20black%20leather%20minidress%20leaning%20doubled%20over%20pressing%20a%20black%20rubber%20funnel%20down%20on%20our%20legs%2C%20ceiling%20light%20behind%2C%20in%20a%20concrete%20dungeon%20looking%20at%20us",
        "https://hotpot.ai/s/art-generator/8-GCdHgPuLegcTS94?share=1&title=girl%20whit%20no%20collthes%20full%20body",
        "https://hotpot.ai/s/art-generator/8-aJvD042KxsyHlRd?share=1&title=ur%20mom%20as%20a%20furry",
        "https://hotpot.ai/s/art-generator/8-kZ6EQT7FmHGETb2?share=1&title=a%20older%20brunette%20woman%20with%20long%20straight%20hair%20wearing%20daisy%20dukes%20crawling%20on%20her%20knees%20hugs%20a%20large%20werewolf",
        "https://hotpot.ai/s/art-generator/8-7TVSo4wC4qEGgRM?share=1&title=Muscled%20Mike%20Shinoda%20at%2036%20wearing%20elegant%20white%20korean%20shirt%20slightly%20smiling%20in%20rome",
        "https://hotpot.ai/s/art-generator/8-2x3HhqUwIfbP3EQ?share=1&title=human%20woman%27s%20body%20with%20a%20cat%27s%20tail%20and%20fur%20growing%20out%20of%20her%20backside%20covering%20her%20whole%20body",
        "https://hotpot.ai/s/art-generator/8-bJtPqU7R0B03rab?share=1&title=hot%20giant%20young%20asian%20tall%20girl%20with%20hide%20hips%20crouching%20on%20the%20floor%20in%20bedroom%20%2C%20show%20full%20body%20from%20the%20back%2C%20tall%20legs%2C%20untied%20hair%2C%20thick%2C%20undresing%2C%20holding%20her%20underwer%2C%20black%20hair%20at%20midnight%2C%20curvy%2C%20view%20from%20below%2C%20hugging%20a%20man%20shorter%20than%20her%2C%20body%20close%20view",
        "https://hotpot.ai/s/art-generator/8-1ypi8ibo6F7C8Zf?share=1&title=Hot%20%7B%7BE%7DROTICS%7DSIN%20City%20BoRDeLlO.%7BSeductiv%7Bely%7D%7DHot-EnchantrESS%2C%20Hot-Provocative%2C%20(%7BNaught%7BYy%7D%7DBlacked%20Diamond-secret%20Ppearl%20Sinful%7BSe%7Bxy%7D%7DHoTt%7BPor%7BNn%7D%7D%20%7BBondag%7BE%7D%7DSlave)Leia%20Skywalker%20Tera%20Patrick%20Ttaimanin%20Asagi%20Lilith%20Soft%20HaREeM%20HoTt-WaifU.%20Hot%20DEepeSt%20RrICH-ToNe%7B%7BSs%7Deducer%7DTreaSurE-ESS%20CoFfee-OiLled%20OiLlY%20%7BSuCcuBu%7BSs%7D%7DSkiN.%7BKeSs18%7B%2B%7D%7DAdults%20Only.%20Strip-Pole%7BKink%7BeSs%7D%7D%7BStripp%7Ber%7D%7DLasciviouseSs",
        "https://hotpot.ai/s/art-generator/8-gdAl6mSaxS34Uin?share=1&title=Two%20abusive%20and%20amorously%20aggresive%20fortyfiveyearold%20masculine%20sjw%20feminist%20queer%20lesbian%20goth%20shemales%20with%20long%20messy%20provocative%20purple%20hair%20with%20straight%20bangs%2C%20thick%20eyebrows%2C%20long%20unshaved%20bodyhair%2C%20and%20thick%20bulging%20camel-toe%20smiling%20predatorally%20and%20trying%20to%20peg%20you%20with%20strapons%20in%20a%20filthy%20pink%20feminine%20lesbian%20bedroom",
        "https://hotpot.ai/s/art-generator/8-jKCrm2qiNwmO8Rq?share=1&title=Two%20abusive%20and%20amorously%20aggresive%20fortyfiveyearold%20masculine%20sjw%20feminist%20queer%20lesbian%20goth%20shemales%20with%20long%20messy%20provocative%20purple%20hair%20with%20straight%20bangs%2C%20thick%20eyebrows%2C%20long%20unshaved%20bodyhair%2C%20and%20thick%20bulging%20camel-toe%20smiling%20predatorally%20and%20trying%20to%20peg%20you%20with%20strapons%20in%20a%20filthy%20pink%20feminine%20lesbian%20bedroom",
        "https://hotpot.ai/s/art-generator/8-qk59fI9ZGQmdWQb?share=1&title=extreme%20low-angle%20extreme%20close-up%20a%20strong%20dark%20powerful%20big%20plump%20sweaty%20italian%20woman%20short%20bobbed%20shoulder-length%20hair%20slightly%20large%20nose%20full%20lips%20black%20leather%20minidress%20squeezing%20a%20man%20in%20an%20iron%20container%20below%20her%2C%20ceiling%20light%20behind%2C%20in%20a%20concrete%20dungeon%20looking%20at%20us",
        "https://hotpot.ai/s/art-generator/8-w0UV05Noo02g0rL?share=1&title=muscular%20curvy%20thick%20serena%20williams%20as%20spider-man%2C%20view%20from%20back",
        "https://hotpot.ai/s/art-generator/8-vqy3LkFRqhNhr0a?share=1&title=full%20body%20photo%3B%20a%20young%20Caucasian%20woman%20with%20a%20confident%20expression%2C%20short%20ponytail%20Auburn%20hairstyle%2C%20freckled%20face.%20she%27s%20wearing%20a%20black%20hoodie%20with%20hot-pink%20yoga%20pants%2C%20and%20small%20necklace.%20Rooftop%20setting.",
        "https://hotpot.ai/s/art-generator/8-7woeoXKaO9pCNfX?share=1&title=extreme%20close-up%20a%20strong%20dark%20powerful%20big%20plump%20sweaty%20italian%20woman%20short%20bobbed%20shoulder-length%20hair%20slightly%20large%20nose%20full%20lips%20black%20leather%20minidress%20turning%20a%20handle%20on%20a%20grimacing%20man%20who%20is%20inside%20a%20long%20iron%20container%20beneath%20her%2C%20in%20a%20concrete%20dungeon",
        "https://hotpot.ai/s/art-generator/8-CoLPQJ72gLWUan3?share=1&title=two%20angry%20women%20with%20beautiful%20bodies%20in%20too%20short%20lambada%20lace%20miniskirts%20and%20protuding%20beak-masks%20grappling%20in%20an%20empty%2019th%20century%20room%2C%20seen%20from%20the%20back",
        "https://hotpot.ai/s/art-generator/8-f2wKqTmvMoySf7K?share=1&title=two%20angry%20women%20with%20beautiful%20bodies%20in%20too%20short%20lambada%20lace%20miniskirts%20and%20protuding%20beak-masks%20grappling%20in%20an%20empty%2019th%20century%20room%2C%20seen%20from%20the%20back",
        "https://hotpot.ai/s/art-generator/8-CtClcWw31Z4ijXy?share=1&title=hot%20giant%20%20young%20tall%20girl%20with%20hide%20hips%20crouching%20on%20the%20floor%20in%20wardrobe%20%2C%20show%20full%20body%20from%20the%20back%2C%20tall%20legs%2C%20untied%20hair%2C%20thick%2C%20undresing%2C%20holding%20her%20underwer%2C%20black%20hair",
    ]

    # Use ThreadPoolExecutor to process links concurrently
    max_workers = min(8, len(test_links))  # Adjust based on your system capabilities
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {executor.submit(detector.is_nsfw, link): link for link in test_links}
        for future in as_completed(future_to_link):
            link = future_to_link[future]
            try:
                result = future.result()
            except Exception as e:
                logging.error(f"Error processing link {link}: {e}")
                continue
            print(f"\nChecking: {link}")
            print(f"Result: {json.dumps(result, indent=2)}")

    # Print detailed report
    print("\nDetailed Processing Report:")
    print(f"Total links processed: {len(test_links)}")
    
    # Count actual files in directories
    sfw_files = len(list(detector.output_dirs['sfw'].glob('*')))
    nsfw_files = len(list(detector.output_dirs['nsfw'].glob('*')))
    print(f"\nFiles saved:")
    print(f"SFW folder: {sfw_files} files")
    print(f"NSFW folder: {nsfw_files} files")
    print(f"Total saved: {sfw_files + nsfw_files} files")
    
    print(f"\nProcessing breakdown:")
    print(f"Actually saved: {sfw_files + nsfw_files}")  # Use actual file count instead
    print(f"Failed: {len(detector.processed_links['failed'])}")
    print(f"Blocked by keywords: {len(detector.processed_links['keyword_blocked'])}")
    print(f"Blocked by text classification: {len(detector.processed_links['text_blocked'])}")
    print(f"Skipped: {len(detector.processed_links['skipped'])}")
    
    # Check for missing file
    if len(test_links) != (sfw_files + nsfw_files):
        print(f"\nWarning: {len(test_links) - (sfw_files + nsfw_files)} link(s) did not result in saved files")
        
        # Add debugging to find the missing link
        saved_filenames = set()
        for folder in [detector.output_dirs['sfw'], detector.output_dirs['nsfw']]:
            for file in folder.glob('*'):
                saved_filenames.add(file.stem)
                
        # Track which links didn't result in files
        for link in test_links:
            # Extract ID from hotpot link
            match = re.search(r'8-[\w\d]+', link)
            if match and match.group(0) not in saved_filenames:
                print(f"\nMissing file for link: {link}")

if __name__ == "__main__":
    main()
