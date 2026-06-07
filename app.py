from flask import Flask, render_template, request, redirect, session, jsonify, flash, url_for
import mysql.connector
from mysql.connector import pooling
import cv2
import numpy as np
import json
import logging
from functools import wraps
from tensorflow.keras.models import load_model
import pickle
from mtcnn import MTCNN

app = Flask(__name__)
app.secret_key = "secret123"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("faceattend")


def handle_errors(response_type: str = "html"):
    """
    Tangkap exception di view, log traceback, kembalikan respons aman ke klien.
    response_type: 'json' | 'plain' | 'html'
    """

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            try:
                return view_func(*args, **kwargs)
            except Exception:
                logger.exception(
                    "Error in view %r — %s %s",
                    view_func.__name__,
                    request.method,
                    request.path,
                )
                if response_type == "json":
                    return jsonify({
                        "error": "internal_error",
                        "message": "Terjadi kesalahan di server.",
                    }), 500
                if response_type == "plain":
                    return "Internal Server Error", 500
                flash("Terjadi kesalahan server.", "danger")
                return redirect(request.referrer or url_for("login"))

        return wrapped

    return decorator

# Konfigurasi Database Pool (Maksimal 20 koneksi simultan)
db_pool = pooling.MySQLConnectionPool(
    pool_name="faceattend_pool",
    pool_size=20,
    pool_reset_session=True,
    host="localhost",
    user="root",
    password="",
    database="db_absensi"
)

def get_db():
    """Mengambil koneksi database dari pool."""
    return db_pool.get_connection()

# ================= LOAD MODEL =================

logger.info("Memuat model face recognition...")

face_model = None
le = None
detector = None
liveness_model = None
le_liveness = None

try:

    # load CNN model
    face_model = load_model(
        'best_model.keras'
    )

    # load label encoder
    with open(
        'label_encoder_cnn_new.pickle',
        'rb'
    ) as f:

        le = pickle.loads(f.read())

    # load MTCNN detector
    detector = MTCNN()

    # load liveness model
    liveness_model = load_model('liveness_model.keras')

    # load liveness label encoder
    with open('Label_encoder_liveness.pickle', 'rb') as f:
        le_liveness = pickle.loads(f.read())

    logger.info("Model berhasil dimuat.")

except Exception as e:

    logger.error("Gagal memuat model: %s", e, exc_info=True)

    face_model = None
    le = None
    detector = None
    liveness_model = None
    le_liveness = None

# ================= FACE RECOGNITION =================

@app.route('/recognize', methods=['POST'])
@handle_errors("json")
def recognize():
    

    if face_model is None or le is None or detector is None or liveness_model is None or le_liveness is None:

        return jsonify({
            "error": "model_unavailable",
            "message": (
                "Model pengenalan wajah atau liveness tidak dapat dimuat. "
                "Pastikan semua file model dan encoder tersedia, lalu restart server."
            ),
        }), 503

    if 'image' not in request.files:

        return jsonify({
            "error": "missing_image",
            "message": "Permintaan tidak menyertakan field gambar (image).",
        }), 400

    file = request.files['image']
    raw = file.read()

    if not raw:

        return jsonify({
            "error": "empty_image",
            "message": "Gambar kosong — kamera mungkin belum siap.",
        }), 400

    nparr = np.frombuffer(
        raw,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:

        return jsonify({
            "nama": "Invalid Image",
            "liveness": "Unknown",
            "confidence": 0
        })

    # BGR -> RGB
    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # DETECT FACE USING MTCNN
    results = detector.detect_faces(rgb)

    # filter confidence
    results = [r for r in results if r['confidence'] >= 0.8]

    if len(results) == 0:

        return jsonify({
            "nama": "No Face",
            "liveness": "Unknown",
            "confidence": 0
        })

    # AMBIL WAJAH TERBESAR
    results = sorted(results, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    result = results[0]

    x, y, w, h = result['box']

    if w <= 0 or h <= 0:

        return jsonify({
            "nama": "Invalid Face",
            "liveness": "Unknown",
            "confidence": 0
        })

    x = max(0, x)
    y = max(0, y)

    # landmark
    try:
        keypoints = result['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        # sudut wajah
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # titik tengah mata
        center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))

        # rotate wajah
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]))
    except:
        aligned = rgb

    # padding wajah
    pad = int(min(w, h) * 0.12)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(aligned.shape[1], x + w + pad)
    y2 = min(aligned.shape[0], y + h + pad)

    # crop wajah
    face = aligned[y1:y2, x1:x2]

    if face.size == 0:

        return jsonify({
            "nama": "Invalid Face",
            "liveness": "Unknown",
            "confidence": 0
        })

    face_raw = face.copy()

    # ================= PREPROCESSING =================

    # CLAHE enhancement
    lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    face_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # resize final
    face = cv2.resize(
        face_enhanced,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    # normalisasi
    face = face.astype("float32") / 255.0

    # reshape RGB
    face = face.reshape(
        1,
        128,
        128,
        3
    )

    # ================= PREDICTION =================

    preds = face_model.predict(
        face,
        verbose=0
    )[0]

    idx = int(np.argmax(preds))

    confidence = float(preds[idx])

    # threshold
    if confidence > 0.85:

        name = le.classes_[idx]

    else:

        name = "Unknown"

    # ================= CNN LIVENESS =================

    try:
        if liveness_model is not None and le_liveness is not None:
            # 1. CLAHE 
            lab = cv2.cvtColor(face_raw, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_clahe = clahe.apply(l)
            lab = cv2.merge((l_clahe, a, b))
            face_liveness = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 2. Resize and normalize
            face_liveness = cv2.resize(face_liveness, (128, 128), interpolation=cv2.INTER_AREA)
            face_liveness = face_liveness.astype("float32") / 255.0
            face_liveness = np.expand_dims(face_liveness, axis=0)
            
            # 3. Predict
            pred_liveness = liveness_model.predict(face_liveness, verbose=0)[0][0]
            
            # 4. Ambil label (0 untuk live, 1 untuk spoof)
            idx_liveness = 0 if pred_liveness < 0.5 else 1
            liveness_label = le_liveness.inverse_transform([idx_liveness])[0].lower()
            
            if liveness_label == 'live':
                liveness = "Real"
            else:
                liveness = "Fake"
        else:
            # Fallback jika model gagal dimuat
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            liveness = "Real" if laplacian_var > 100 else "Fake"
            
    except Exception as e:
        logger.error(f"Error pada liveness detection: {e}", exc_info=True)
        liveness = "Unknown"

    return jsonify({
        "nama": name,
        "liveness": liveness,
        "confidence": confidence * 100
    })

# Route Login
@app.route('/', methods=['GET', 'POST'])
@handle_errors("html")
def login():
    if request.method == 'POST':
        nim = request.form['nim']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE nim=%s AND password=%s", (nim, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            session['nim'] = user['nim']
            session['nama'] = user['nama']
            session['role'] = user['role']
            return redirect('/admin' if user['role'] == 'admin' else '/peserta')
        else:
            flash("Login gagal! Cek kembali NIM dan password.", "danger")
            return redirect('/')

    return render_template('login.html')

# Halaman Dashboard Admin
@app.route('/admin')
@handle_errors("html")
def admin():
    if 'role' not in session or session['role'] != 'admin':
        return "Akses ditolak!"

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM absensi WHERE DATE(waktu_masuk) = CURDATE() ORDER BY waktu_masuk DESC")
    data = cursor.fetchall()

    hadir_count = len([d for d in data if d.get('status') == 'hadir'])
    spoof_count = len([d for d in data if d.get('status') == 'tidak_hadir'])

    cursor.execute("SELECT COUNT(*) AS total FROM peserta")
    total_mahasiswa = cursor.fetchone()['total']
    
    cursor.close()
    conn.close()

    return render_template('admin.html', data=data, total_mahasiswa=total_mahasiswa, 
                           hadir_count=hadir_count, spoof_count=spoof_count)

# Halaman Manajemen Peserta
@app.route('/data-peserta')
@handle_errors("html")
def data_peserta():
    if 'role' not in session or session['role'] != 'admin':
        return "Akses ditolak!"

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM peserta")
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('data_peserta.html', data=data)

# Halaman Peserta & Video Conference
@app.route('/peserta')
@handle_errors("html")
def peserta():
    if 'role' not in session or session['role'] != 'peserta':
        return "Akses ditolak!"
    return render_template('peserta.html', nama=session['nama'])

# Kelola Peserta (Tambah/Edit/Hapus)
@app.route('/tambah-peserta', methods=['POST'])
@handle_errors("html")
def tambah_peserta():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO peserta (nim, nama, password) VALUES (%s, %s, %s)", 
                   (request.form['nim'], request.form['nama'], request.form['password']))
    cursor.execute("INSERT INTO users (nim, nama, password, role) VALUES (%s, %s, %s, 'peserta')", 
                   (request.form['nim'], request.form['nama'], request.form['password']))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect('/data-peserta')

@app.route('/edit-peserta/<nim>', methods=['POST'])
@handle_errors("html")
def edit_peserta(nim):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE peserta SET nama=%s, password=%s WHERE nim=%s", 
                   (request.form['nama'], request.form['password'], nim))
    cursor.execute("UPDATE users SET nama=%s, password=%s WHERE nim=%s", 
                   (request.form['nama'], request.form['password'], nim))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect('/data-peserta')

@app.route('/hapus-peserta/<nim>')
@handle_errors("html")
def hapus_peserta(nim):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM peserta WHERE nim=%s", (nim,))
    cursor.execute("DELETE FROM users WHERE nim=%s", (nim,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect('/data-peserta')

# Catat Absen Masuk
@app.route('/absen', methods=['POST'])
@handle_errors("plain")
def absen():

    if 'role' not in session or session['role'] != 'peserta':
        return "Akses ditolak!"

    nim = session.get('nim')
    nama = session.get('nama')
    status = request.form.get('status')
    keterangan = request.form.get('keterangan', 'Real')

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Cek apakah peserta masih aktif
    cursor.execute("""
        SELECT * FROM absensi
        WHERE nim = %s
        AND DATE(waktu_masuk) = CURDATE()
        AND waktu_keluar IS NULL
        LIMIT 1
    """, (nim,))

    existing = cursor.fetchone()

    # Jika belum ada absensi aktif -> INSERT
    if not existing:

        insert_cursor = conn.cursor()

        insert_cursor.execute("""
            INSERT INTO absensi
            (nim, nama, status, keterangan)
            VALUES (%s, %s, %s, %s)
        """, (nim, nama, status, keterangan))

        conn.commit()

        insert_cursor.close()

    cursor.close()
    conn.close()

    return "OK"

# Rekap Data Absensi
@app.route('/data-absensi', methods=['GET', 'POST'])
@handle_errors("html")
def data_absensi():
    if 'role' not in session or session['role'] != 'admin':
        return "Akses ditolak!"

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    if request.method == 'POST':
        cursor.execute("SELECT * FROM absensi WHERE DATE(waktu_masuk) = %s ORDER BY waktu_masuk DESC", 
                       (request.form.get('tanggal'),))
    else:
        cursor.execute("SELECT * FROM absensi ORDER BY waktu_masuk DESC")
    
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('data_absensi.html', data=data)

# Catat Waktu Keluar
@app.route('/keluar', methods=['POST'])
@handle_errors("plain")
def keluar():

    nim = session.get('nim')

    if not nim:
        return "NIM tidak ditemukan"

    status = request.form.get('status')
    keterangan = request.form.get('keterangan')

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE absensi
        SET 
            waktu_keluar = CURRENT_TIMESTAMP,
            status = %s,
            keterangan = %s
        WHERE nim = %s
        AND waktu_keluar IS NULL
        ORDER BY waktu_masuk DESC
        LIMIT 1
    """, (status, keterangan, nim))

    conn.commit()

    cursor.close()
    conn.close()

    return "OK"

@app.route('/logout')
@handle_errors("html")
def logout():
    session.clear()
    return redirect('/')

# Jalankan Flask (Threaded agar support banyak user sekaligus)
if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)