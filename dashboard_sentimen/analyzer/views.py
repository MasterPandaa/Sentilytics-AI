# analyzer/views.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
import pickle # <--- TAMBAHAN PENTING

from django.shortcuts import render, redirect
from django.http import HttpResponse # <--- Untuk download file
from .forms import UploadFileForm

# Sklearn & Metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- FUNGSI BANTUAN ---

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_graph():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', transparent=True)
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close()
    return graph

def get_star_rating(score):
    stars = round((score / 100) * 5, 1)
    full_stars = int(stars)
    has_half = (stars - full_stars) >= 0.5
    empty_stars = 5 - full_stars - (1 if has_half else 0)
    
    return {
        'value': stars,
        'full': list(range(full_stars)),
        'half': has_half,
        'empty': list(range(empty_stars))
    }

def truncate_name(name, length=18):
    if len(name) > length:
        return name[:length] + "..."
    return name

def analyze_single_dataset(df, filename, n_estimators=50, sample_size=5000):
    
    # 1. Labeling
    def get_label(r):
        return 'Negative' if r < 3 else ('Neutral' if r == 3 else 'Positive')
    df['Label_Asli'] = df['Rating'].apply(get_label)
    
    # 2. Sampling
    real_count = len(df)
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df.copy()
        
    df_sample['Clean_Text'] = df_sample['Ulasan'].apply(clean_text)
    
    # 3. Training
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df_sample['Clean_Text'])
    y = df_sample['Label_Asli']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 4. Metrics
    classes = model.classes_
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 5. Sentimen Counts
    sent_counts = df['Label_Asli'].value_counts().to_dict()
    total_sent = sum(sent_counts.values())
    pos_pct = round((sent_counts.get('Positive', 0) / total_sent) * 100, 1)
    neg_pct = round((sent_counts.get('Negative', 0) / total_sent) * 100, 1)
    
    # --- XAI & NARRATIVE ---
    feature_names = tfidf.get_feature_names_out()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_words = [feature_names[i] for i in indices[:5]]
    top_words_str = ", ".join([f"<b>'{w}'</b>" for w in top_words])
    
    verdict = ""
    if pos_pct > 65:
        verdict = "Favorit Pengguna 😍"
        verdict_color = "success"
        intro = f"Mayoritas pengguna merasa puas ({pos_pct}%)."
    elif neg_pct > 65:
        verdict = "Krisis Reputasi 🚨"
        verdict_color = "danger"
        intro = f"Terdapat gelombang keluhan yang signifikan ({neg_pct}%)."
    elif pos_pct > neg_pct:
        verdict = "Cukup Positif 🙂"
        verdict_color = "info"
        intro = "Respon pengguna cenderung positif namun ada catatan."
    else:
        verdict = "Perlu Evaluasi ⚠️"
        verdict_color = "warning"
        intro = "Sentimen negatif sedikit lebih dominan."

    if acc > 0.85:
        ai_insight = f"Model AI sangat yakin dengan pola ulasan ini. Faktor penentu utama sentimen adalah penggunaan kata: {top_words_str}."
    elif acc > 0.70:
        ai_insight = f"AI mendeteksi pola yang cukup konsisten. Isu utama berpusat pada topik: {top_words_str}."
    else:
        ai_insight = f"Pola ulasan cukup acak/beragam. Namun, AI menyoroti kata kunci: {top_words_str} sebagai pembeda."

    narrative = f"{intro} <br><br> {ai_insight}"

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax, annot_kws={"size": 12})
    ax.set_title(f'Confusion Matrix: {filename}', fontsize=14)
    cm_image = get_graph()

    # --- 7. EXPORT MODEL TO PICKLE (BASE64) ---
    # Kita simpan Model DAN Vectorizer agar siap pakai
    model_pipeline = {
        'model': model,
        'vectorizer': tfidf,
        'info': f"Model trained on {filename} with accuracy {acc:.2f}"
    }
    
    # Serialize ke bytes lalu ke base64 string
    buffer = io.BytesIO()
    pickle.dump(model_pipeline, buffer)
    buffer.seek(0)
    model_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # -------------------------------------------

    return {
        'name': truncate_name(filename),
        'full_name': filename,
        'total_data': real_count,
        'used_sample': len(df_sample),
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'cm_image': cm_image,
        'sentiment_counts': sent_counts,
        'stars': get_star_rating(acc * 100),
        'narrative': narrative,
        'verdict': verdict,
        'verdict_color': verdict_color,
        'model_file': model_b64 # Data Model tersimpan di sini
    }

def generate_comparison_bar(results):
    names = [r['name'] for r in results]
    positives = [r['sentiment_counts'].get('Positive', 0) for r in results]
    neutrals = [r['sentiment_counts'].get('Neutral', 0) for r in results]
    negatives = [r['sentiment_counts'].get('Negative', 0) for r in results]
    
    x = np.arange(len(names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    rects1 = ax.bar(x - width, positives, width, label='Positive', color='#2ecc71')
    rects2 = ax.bar(x, neutrals, width, label='Neutral', color='#95a5a6')
    rects3 = ax.bar(x + width, negatives, width, label='Negative', color='#e74c3c')
    
    ax.set_ylabel('Jumlah Ulasan')
    ax.set_title('Perbandingan Distribusi Sentimen')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight='bold')
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
        
    return get_graph()

def reset_session(request):
    if 'stored_results' in request.session:
        del request.session['stored_results']
    return redirect('dashboard')

# --- VIEW BARU UNTUK DOWNLOAD ---
def download_model(request, index):
    stored_results = request.session.get('stored_results', [])
    
    # Validasi index
    if 0 <= index < len(stored_results):
        result = stored_results[index]
        model_b64 = result.get('model_file')
        filename = result.get('full_name', 'model').replace('.csv', '')
        
        if model_b64:
            # Decode base64 kembali ke bytes
            model_bytes = base64.b64decode(model_b64)
            
            # Buat HTTP Response untuk file download
            response = HttpResponse(model_bytes, content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="model_{filename}.pkl"'
            return response
    
    return redirect('dashboard')

def dashboard_view(request):
    context = {}
    stored_results = request.session.get('stored_results', [])
    form = UploadFileForm()
    
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            n_est = form.cleaned_data['n_estimators']
            sample_size = form.cleaned_data['sample_size']
            
            try:
                df = pd.read_csv(file)
                cols = [c.lower() for c in df.columns]
                if 'ulasan' in cols and 'rating' in cols:
                    df.columns = [c.capitalize() for c in df.columns]
                    
                    clean_name = file.name.replace('ulasan_', '').replace('.csv', '').replace('com.', '').title()
                    
                    new_result = analyze_single_dataset(df, clean_name, n_est, sample_size)
                    
                    if len(stored_results) >= 3:
                        stored_results.pop(0)
                    
                    stored_results.append(new_result)
                    request.session['stored_results'] = stored_results
                    request.session.modified = True
                else:
                    context['error'] = "Format salah. Dataset WAJIB memiliki kolom 'Ulasan' dan 'Rating'."
            except Exception as e:
                context['error'] = f"Gagal memproses file: {str(e)}"

    if stored_results:
        context['comparison_chart'] = generate_comparison_bar(stored_results)
        context['results'] = stored_results
        
    context['form'] = form
    return render(request, 'analyzer/index.html', context)

# --- FUNGSI BARU UNTUK DOWNLOAD ---
def download_model(request, index):
    stored_results = request.session.get('stored_results', [])
    
    # Validasi index
    if 0 <= index < len(stored_results):
        result = stored_results[index]
        model_b64 = result.get('model_file')
        filename = result.get('full_name', 'model').replace('.csv', '')
        
        if model_b64:
            # Decode base64 kembali ke bytes
            model_bytes = base64.b64decode(model_b64)
            
            # Buat HTTP Response untuk file download
            response = HttpResponse(model_bytes, content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="model_{filename}.pkl"'
            return response
    
    return redirect('dashboard')