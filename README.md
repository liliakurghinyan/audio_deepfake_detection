# Audio Deepfake Detection System

Ձայնային deepfake-ների հայտնաբերման համակարգ՝ հիմնված **Mel Spectrogram feature extraction** և **Convolutional Neural Network (CNN)** մոդելի վրա։  
Համակարգը նախատեսված է իրական և AI-գեներացված հայերեն ձայների տարբերակման համար։

---

## 📌 Հիմնական հնարավորություններ

- WAV, MP3, FLAC, M4A և այլ ֆորմատների աջակցություն
- Ավտոմատ audio conversion դեպի WAV
- Mel Spectrogram feature extraction
- CNN-based classification
- Web interface (FastAPI + HTML/CSS)
- Real/Fake probability prediction
- Validation/Test metrics visualization

---

## 🛠 Օգտագործված տեխնոլոգիաներ

- Python
- PyTorch
- Torchaudio
- FastAPI
- Librosa
- SoundFile
- Pydub
- HTML/CSS

---

## 📁 Project Structure

```bash
dataset/
web/
results/
train.py
evaluate.py
infer.py
detector.py
app.py
model.py
features.py
