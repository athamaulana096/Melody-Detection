from flask import Flask, request, render_template, jsonify, send_file
import os
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Disable interactive mode
import matplotlib.pyplot as plt
import librosa.display
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STEMS_FOLDER = 'stems'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STEMS_FOLDER, exist_ok=True)

N_FFT = 1024
segment_times = []  # Define segment_times globally
predictions = []  # Define predictions globally
uploaded_file_path = None  # Define uploaded_file_path globally

def extract_mel_spectrogram(signal, sr=22050, n_fft=N_FFT, n_mels=128, target_size=(128, 128)):
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')  # Padding untuk sinyal pendek
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    if mel_spectrogram_db.shape[1] < target_size[1]:
        padding = target_size[1] - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), mode='constant')
    elif mel_spectrogram_db.shape[1] > target_size[1]:
        mel_spectrogram_db = mel_spectrogram_db[:, :target_size[1]]
    return mel_spectrogram_db

def detect_pitch_yin(signal, sr=22050, fmin=50, fmax=500, target_size=128):
    if len(signal) < N_FFT:
        signal = np.pad(signal, (0, N_FFT - len(signal)), mode='constant')  # Padding untuk sinyal pendek
    pitches, magnitudes = librosa.core.pitch.piptrack(y=signal, sr=sr, n_fft=N_FFT, fmin=fmin, fmax=fmax)
    pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1])]
    pitch_values = np.array(pitch_values)
    if pitch_values.max() != pitch_values.min():
        pitch_values = (pitch_values - pitch_values.min()) / (pitch_values.max() - pitch_values.min())
    else:
        pitch_values = np.zeros(target_size)
    if len(pitch_values) < target_size:
        padding = target_size - len(pitch_values)
        pitch_values = np.pad(pitch_values, (0, padding), mode='constant')
    elif len(pitch_values) > target_size:
        pitch_values = pitch_values[:target_size]
    return pitch_values

def process_audio_segments_by_transient(file_path, sr=22050, segment_duration=10):
    signal, sr = librosa.load(file_path, sr=sr)
    total_duration = len(signal) / sr
    
    # Calculate number of full segments needed
    segment_length = segment_duration * sr
    num_segments = max(1, int(np.ceil(total_duration / segment_duration)))
    
    segments = []
    segment_times = []

    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(signal))
        segment = signal[start:end]

        # Only process if segment has sufficient length
        if len(segment) > N_FFT:
            # Process by transient within each segment
            transients = librosa.onset.onset_detect(y=segment, sr=sr, units='samples')
            
            # If no transients detected, treat the whole segment as one
            if len(transients) == 0:
                mel_spectrogram = extract_mel_spectrogram(segment, sr)
                pitch_values = detect_pitch_yin(segment, sr)
                segments.append((mel_spectrogram, pitch_values))
                segment_times.append(start / sr)
            else:
                # Add start and end points
                transients = np.concatenate([[0], transients, [len(segment)]])
                
                for j in range(len(transients) - 1):
                    t_start = transients[j]
                    t_end = transients[j + 1]
                    
                    # Only process if sub-segment is long enough
                    if t_end - t_start > N_FFT:
                        transient_segment = segment[t_start:t_end]
                        mel_spectrogram = extract_mel_spectrogram(transient_segment, sr)
                        pitch_values = detect_pitch_yin(transient_segment, sr)
                        segments.append((mel_spectrogram, pitch_values))
                        segment_times.append((start + t_start) / sr)

    return segments, segment_times

def predict_segments(model, segments, segment_times, threshold=0.3):
    labels = ['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Si']
    all_predictions = []

    for (mel_spectrogram, pitch_values), start_time in zip(segments, segment_times):
        mel_spectrogram = mel_spectrogram[..., np.newaxis]
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        pitch_values = np.expand_dims(pitch_values, axis=0)

        prediction = model.predict([mel_spectrogram, pitch_values])
        for i, val in enumerate(prediction[0]):
            if val > threshold:
                all_predictions.append({
                    'note': labels[i],
                    'time': float(start_time),
                    'confidence': float(val),
                    'duration': 0.5  # Default duration
                })

    return all_predictions

def save_audio_segments(file_path, output_folder, sr=22050, segment_duration=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    signal, sr = librosa.load(file_path, sr=sr)
    segment_length = segment_duration * sr
    num_segments = int(np.ceil(len(signal) / segment_length))

    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(signal))
        segment = signal[start:end]
        output_file = os.path.join(output_folder, f'segment{i+1}.wav')
        sf.write(output_file, segment, sr)
        print(f'Saved {output_file}')

def plot_predictions(file_path, predictions, sr=22050, segment_duration=10, segment_index=0):
    signal, sr = librosa.load(file_path, sr=sr)
    segment_length = segment_duration * sr
    start = segment_index * segment_length
    end = min((segment_index + 1) * segment_length, len(signal))
    segment_signal = signal[start:end]

    # Debug prints
    print(f"Plotting segment {segment_index}")
    print(f"Start time: {start/sr:.2f}s")
    print(f"End time: {end/sr:.2f}s")
    print(f"All predictions: {predictions}")

    # Filter predictions for this segment
    segment_predictions = []
    for pred in predictions:
        pred_time = float(pred['time'])
        if start/sr <= pred_time < end/sr:
            segment_predictions.append(pred)
            print(f"Including prediction {pred} in segment {segment_index}")

    # Pad if needed
    if len(segment_signal) < segment_length:
        segment_signal = np.pad(segment_signal, (0, segment_length - len(segment_signal)), mode='constant')

    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot waveform
    librosa.display.waveshow(segment_signal, sr=sr, alpha=0.5, ax=ax)
    
    # Plot predictions with clearer markers
    for pred in segment_predictions:
        label = pred['note']
        time_abs = pred['time']
        time_relative = time_abs - (start / sr)
        ax.axvline(x=time_relative, color='red', alpha=0.8, ymin=0.4, ymax=0.6, linestyle='--')
        ax.text(time_relative, max(segment_signal) * 1.2, label, color='r')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Segment {segment_index + 1} - Audio Waveform with Note Predictions")
    ax.grid(True, alpha=0.3)

    # Save plot
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def split_stems(file_path, output_folder='stems'):
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    model = get_model('htdemucs')
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
    
    # Load audio
    audio, sr = librosa.load(file_path, sr=44100, mono=False)
    if audio.ndim == 1:
        audio = audio[None]  # Add a channel dim
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)  # Convert mono to stereo
    
    # Convert to torch tensor
    audio = torch.tensor(audio)
    
    # Apply model
    with torch.no_grad():
        sources = apply_model(model, audio[None], device='cuda' if torch.cuda.is_available() else 'cpu')[0]
    
    # Save stems
    stem_paths = {}
    sources = sources.cpu().numpy()
    for source, name in zip(sources, model.sources):
        stem_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_{name}.wav")
        sf.write(stem_path, source.T, sr)
        stem_paths[name] = stem_path
    
    return stem_paths

@app.route('/')
def index():
    return render_template('index.html', segment_times=segment_times)

@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_file_path
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
            
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        uploaded_file_path = file_path
        
        return jsonify({'success': True, 'filename': file.filename})
        
    except Exception as e:
        print(f"Upload route error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    global segment_times, predictions, uploaded_file_path
    
    try:
        # Check if we're getting data from the form or if we should use the uploaded file
        if request.content_type and 'multipart/form-data' in request.content_type and 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                if not uploaded_file_path:
                    return jsonify({'error': 'No selected file'})
            else:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                uploaded_file_path = file_path
        
        if not uploaded_file_path or not os.path.exists(uploaded_file_path):
            return jsonify({'error': 'No file has been uploaded'})
        
        # Get audio duration
        signal, sr = librosa.load(uploaded_file_path, sr=22050)
        duration = len(signal) / sr
        segment_duration = 10  # seconds per segment
        
        print(f"Processing file: {uploaded_file_path}")
        print(f"Audio duration: {duration:.2f} seconds")
        
        segments, segment_times = process_audio_segments_by_transient(uploaded_file_path)
        print(f"Segments processed: {len(segments)}")
        
        if not os.path.exists('model7.h5'):
            return jsonify({'error': 'Model file not found'})
            
        model = load_model('model7.h5')
        predictions = predict_segments(model, segments, segment_times)
        print(f"Final predictions: {predictions}")
        
        # Format segments for frontend display
        formatted_segments = []
        num_segments = max(1, int(np.ceil(duration / segment_duration)))
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            formatted_segments.append({
                'index': i,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            })
        
        return jsonify({
            'segmentTimes': [seg['start'] for seg in formatted_segments],
            'predictions': [(pred['note'], pred['time']) for pred in predictions],
            'totalDuration': duration,
            'numSegments': num_segments
        })
        
    except Exception as e:
        print(f"Prediction route error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/split_stems', methods=['POST'])
def process_stems():
    global uploaded_file_path
    if not uploaded_file_path:
        return jsonify({'error': 'No file uploaded yet'})
    
    try:
        stem_paths = split_stems(uploaded_file_path)
        return jsonify({
            'stems': list(stem_paths.keys()),
            'message': 'Stems split successfully'
        })
    except Exception as e:
        print(f"Stem splitting error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/predict_stem', methods=['POST'])
def predict_stem():
    global segment_times, predictions, uploaded_file_path
    
    data = request.get_json()
    stem_name = data.get('stem')
    if not stem_name:
        return jsonify({'error': 'No stem specified'})
    
    stem_path = os.path.join('stems', f"{os.path.splitext(os.path.basename(uploaded_file_path))[0]}_{stem_name}.wav")
    if not os.path.exists(stem_path):
        return jsonify({'error': f'Stem file not found: {stem_path}'})
    
    try:
        # Use the stem file path instead of the original file
        temp_file_path = uploaded_file_path
        uploaded_file_path = stem_path
        
        # Get audio duration
        signal, sr = librosa.load(stem_path, sr=22050)
        duration = len(signal) / sr
        segment_duration = 10  # seconds per segment
        
        segments, segment_times = process_audio_segments_by_transient(stem_path)
        
        model = load_model('model7.h5')
        predictions = predict_segments(model, segments, segment_times)
        
        # Format segments for frontend display
        formatted_segments = []
        num_segments = max(1, int(np.ceil(duration / segment_duration)))
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            formatted_segments.append({
                'index': i,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            })
        
        # Restore the original file path
        uploaded_file_path = temp_file_path
        
        return jsonify({
            'segmentTimes': [seg['start'] for seg in formatted_segments],
            'predictions': [(pred['note'], pred['time']) for pred in predictions],
            'totalDuration': duration,
            'numSegments': num_segments,
            'currentStem': stem_name
        })
        
    except Exception as e:
        print(f"Stem prediction error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/get_stem_audio/<stem_name>')
def get_stem_audio(stem_name):
    global uploaded_file_path
    if not uploaded_file_path:
        return jsonify({'error': 'No file uploaded yet'})
    
    stem_path = os.path.join(STEMS_FOLDER, f"{os.path.splitext(os.path.basename(uploaded_file_path))[0]}_{stem_name}.wav")
    if not os.path.exists(stem_path):
        return jsonify({'error': f'Stem file not found: {stem_path}'})
    
    return send_file(stem_path, mimetype='audio/wav')

@app.route('/get_original_audio')
def get_original_audio():
    global uploaded_file_path
    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        return jsonify({'error': 'No file uploaded'})
    
    return send_file(uploaded_file_path)

@app.route('/get_segments', methods=['GET'])
def get_segments():
    return jsonify({'segmentTimes': segment_times})

@app.route('/get_segment_image', methods=['GET'])
def get_segment_image():
    global uploaded_file_path
    index = int(request.args.get('index'))
    print(f'Getting image for segment {index}')
    img_base64 = plot_predictions(uploaded_file_path, predictions, segment_index=index)
    print(f'Generated image for segment {index}')
    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)