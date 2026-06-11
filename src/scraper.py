import pytse_client as tse
import os

SYMBOLS = {
    "Foolad": "فولاد",
    "Khodro": "خودرو",
    "Shasta": "شستا",
    "Vbmelat": "وبملت",
    "Fars": "فارس",
    "Zob": "ذوب",
    "Kegel": "کگل",
    "Shabdar": "شبندر",
    "Vasandogh": "وصندوق",
    "Simorgh": "سیمرغ"
}

def download_data():
    os.makedirs("data/raw", exist_ok=True)
    
    for eng_name, fa_symbol in SYMBOLS.items():
        print(f"Downloading {eng_name}...")
        try:
            tse.download(
                symbols=fa_symbol,
                write_to_csv=True,
                base_path="data/raw"
            )
            # rename فارسی به انگلیسی
            fa_path = f"data/raw/{fa_symbol}.csv"
            eng_path = f"data/raw/{eng_name}.csv"
            if os.path.exists(fa_path):
                os.rename(fa_path, eng_path)
            print(f"[OK] {eng_name} saved")
        except Exception as e:
            print(f"[ERROR] {eng_name}: {e}")

if __name__ == "__main__":
    download_data()
