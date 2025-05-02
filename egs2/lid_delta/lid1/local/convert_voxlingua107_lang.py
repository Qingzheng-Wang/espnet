import os
import random
import re
import sys

import pycountry

folder_to_language = {
    "ab": "Abkhazian",
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gn": "Guarani",
    "gu": "Gujarati",
    "gv": "Manx",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "ia": "Interlingua",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "iw": "Hebrew",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Panjabi",
    "pl": "Polish",
    "ps": "Pushto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sco": "Scots",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "war": "Waray",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Mandarin Chinese",
}


def generate_data_files(
    target_root,
    original_root
):

    for x in ["train", "test"]:
        # output file directory
        text_out = os.path.join(target_root, x, "text")
        utt2spk_out = os.path.join(target_root, x, "utt2spk")
        scp_out = os.path.join(target_root, x, "wav.scp")

        os.makedirs(
            os.path.join(target_root, x),
            exist_ok=True,
        )

        # genrate wav.scp
        with open(scp_out, "w") as scp_out:
            for folder in os.listdir(os.path.join(original_root, x)):
                if os.path.isdir(os.path.join(original_root, x, folder)):
                    for file in os.listdir(os.path.join(original_root, x, folder)):
                        scp_out.write(f"{file} {os.path.join(original_root, x, folder, file)}\n")
                else:
                    raise ValueError(f"Folder {folder} is not a directory")

        # genrate text
        folder_to_iso3 = {}
        with open(text_out, "w") as text_out:
            for folder in os.listdir(os.path.join(original_root, x)):
                if folder not in folder_to_language:
                    raise ValueError(f"Folder {folder} not found in folder_to_language")
                language = folder_to_language[folder]
                try:
                    lang_data = pycountry.languages.lookup(language)
                    iso3_code = lang_data.alpha_3  # 
                    folder_to_iso3[folder] = iso3_code
                except LookupError:
                    folder_to_iso3[folder] = "N/A"  # 
                for file in os.listdir(os.path.join(original_root, x, folder)):
                    text_out.write(f"{file} {folder_to_iso3[folder]}\n")
        
        # genrate utt2spk
        with open(utt2spk_out, "w") as utt2spk_out:
            for folder in os.listdir(os.path.join(original_root, x)):
                for file in os.listdir(os.path.join(original_root, x, folder)):
                    utt2spk_out.write(f"{file} {folder}\n")
       


if __name__ == "__main__":
    original_root = "downloads"
    target_root = "data"
    generate_data_files(
        target_root,
        original_root
    )