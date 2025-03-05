from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
import time
import requests
import zipfile
import os
import re
import tempfile
import shutil
import hashlib
import json
from datetime import datetime
import io
import base64
import PyPDF2
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.secret_key = "p_convert_2025_secure_key"  # Thay đổi thành một key bảo mật hơn trong sản phẩm thực tế
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload size

# Tạo thư mục uploads và output nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# URL của file users.json và check-convert trên GitHub
USERS_FILE_URL = "https://raw.githubusercontent.com/thayphuctoan/pconvert/refs/heads/main/user.json"
ACTIVATION_FILE_URL = "https://raw.githubusercontent.com/thayphuctoan/pconvert/main/check-convert"

# Cache các dữ liệu từ GitHub
users_cache = {"data": None, "timestamp": 0}
activation_cache = {"data": None, "timestamp": 0}

# Kết quả xử lý và tiến trình
processing_tasks = {}

# ------------------------------ Utility Functions ------------------------------
def get_users():
    """Lấy danh sách người dùng từ GitHub với cache 5 phút"""
    current_time = time.time()
    if users_cache["data"] is None or current_time - users_cache["timestamp"] > 300:
        try:
            response = requests.get(USERS_FILE_URL)
            if response.status_code == 200:
                users_cache["data"] = json.loads(response.text)
                users_cache["timestamp"] = current_time
                return users_cache["data"]
            else:
                return {}
        except Exception as e:
            print(f"Lỗi khi lấy danh sách người dùng: {str(e)}")
            return {}
    return users_cache["data"]

def get_activated_ids():
    """Lấy danh sách ID đã kích hoạt từ GitHub với cache 5 phút"""
    current_time = time.time()
    if activation_cache["data"] is None or current_time - activation_cache["timestamp"] > 300:
        try:
            response = requests.get(ACTIVATION_FILE_URL)
            if response.status_code == 200:
                activation_cache["data"] = response.text.strip().split('\n')
                activation_cache["timestamp"] = current_time
                return activation_cache["data"]
            else:
                return []
        except Exception as e:
            print(f"Lỗi khi lấy danh sách ID kích hoạt: {str(e)}")
            return []
    return activation_cache["data"]

def authenticate_user(username, password):
    """Xác thực người dùng"""
    users = get_users()
    if username in users and users[username] == password:
        return True
    return False

def generate_hardware_id(username):
    """Tạo hardware ID cố định từ username"""
    hardware_id = hashlib.md5(username.encode()).hexdigest().upper()
    formatted_id = '-'.join([hardware_id[i:i+8] for i in range(0, len(hardware_id), 8)])
    return formatted_id + "-Premium"

def check_activation(hardware_id):
    """Kiểm tra kích hoạt"""
    activated_ids = get_activated_ids()
    return hardware_id in activated_ids

def split_pdf(input_pdf_data, pages_per_part=5):
    """Tách PDF thành nhiều phần"""
    parts = []
    reader = PyPDF2.PdfReader(io.BytesIO(input_pdf_data))
    total_pages = len(reader.pages)
    
    for start in range(0, total_pages, pages_per_part):
        writer = PyPDF2.PdfWriter()
        for i in range(start, min(start + pages_per_part, total_pages)):
            writer.add_page(reader.pages[i])
        
        output_buffer = io.BytesIO()
        writer.write(output_buffer)
        output_buffer.seek(0)
        
        parts.append({
            "name": f"Part {start//pages_per_part + 1} (Pages {start+1}-{min(start+pages_per_part, total_pages)})",
            "data": output_buffer.getvalue()
        })
    
    return parts

def get_timeout(file_size=None):
    """Lấy giá trị timeout dựa trên kích thước file"""
    if file_size is not None:
        if file_size < 5 * 1024 * 1024:  # 5MB
            return (10, 30)
    return (10, 180)

def download_and_read_full_md(zip_url, task_id):
    """Tải và trích xuất nội dung markdown từ file zip"""
    try:
        # Tạo thư mục tạm thời để trích xuất
        temp_dir = tempfile.mkdtemp()
        
        # Tải file zip
        timeout_val = get_timeout()
        resp = requests.get(zip_url, timeout=timeout_val)
        
        if resp.status_code != 200:
            return f"Lỗi: Tải ZIP thất bại. HTTP {resp.status_code}"
        
        # Lưu file zip
        zip_path = os.path.join(temp_dir, "output.zip")
        with open(zip_path, 'wb') as f:
            f.write(resp.content)
        
        # Trích xuất vào thư mục cố định
        extract_dir = os.path.join(app.config['OUTPUT_FOLDER'], f"task_{task_id}")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            md_file_name = None
            for f_name in zip_ref.namelist():
                if f_name.endswith(".md"):
                    md_file_name = f_name
                    break
            
            if not md_file_name:
                shutil.rmtree(temp_dir)
                return "Lỗi: Không tìm thấy file .md trong gói kết quả!"
            
            # Trích xuất vào thư mục cố định
            zip_ref.extractall(extract_dir)
        
        # Đọc file markdown
        md_path = os.path.join(extract_dir, md_file_name)
        with open(md_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # Xóa thư mục tạm
        shutil.rmtree(temp_dir)
        
        # Cập nhật đường dẫn hình ảnh
        md_text = update_md_image_paths(md_text, task_id)
        
        return md_text
    
    except Exception as e:
        return f"Lỗi: Không thể xử lý file ZIP: {str(e)}"

def update_md_image_paths(md_text, task_id):
    """Cập nhật đường dẫn hình ảnh trong markdown"""
    def replace_path(match):
        img_path = match.group(2)
        if img_path.startswith('images/'):
            image_name = os.path.basename(img_path)
            # Thay đổi đường dẫn để trỏ đến API images
            return f'![{match.group(1)}](/images/{task_id}/{image_name})'
        return match.group(0)
    
    pattern = r'!\[(.*?)\]\((images/[^)]+)\)'
    return re.sub(pattern, replace_path, md_text)

def convert_tables_to_md(text):
    """Chuyển đổi bảng HTML sang định dạng markdown"""
    def html_table_to_markdown(table_tag):
        rows = table_tag.find_all('tr')
        max_cols = 0
        for row in rows:
            count_cols = 0
            for cell in row.find_all(['td', 'th']):
                colspan = int(cell.get('colspan', 1))
                count_cols += colspan
            max_cols = max(max_cols, count_cols)
        
        grid = [["" for _ in range(max_cols)] for _ in range(len(rows))]
        
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(['td', 'th']):
                while col_idx < max_cols and grid[row_idx][col_idx] != "":
                    col_idx += 1
                
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                cell_text = cell.get_text(strip=True)
                
                grid[row_idx][col_idx] = cell_text
                
                for r in range(row_idx, row_idx + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r == row_idx and c == col_idx:
                            continue
                        grid[r][c] = ""
                
                col_idx += colspan
        
        md_lines = []
        header_rows = 1
        
        for hr in range(header_rows):
            md_lines.append("| " + " | ".join(grid[hr]) + " |")
        
        align_line = "| " + " | ".join(["---"] * max_cols) + " |"
        md_lines.insert(header_rows, align_line)
        
        for row_idx in range(header_rows, len(rows)):
            md_lines.append("| " + " | ".join(grid[row_idx]) + " |")
        
        return "\n".join(md_lines)
    
    if '<html>' in text and '<table' in text:
        html_parts = text.split('</html>')
        final_text = text
        
        for part in html_parts:
            if '<html>' in part and '<table' in part:
                html_chunk = part[part.find('<html>'):]
                if not html_chunk.endswith('</html>'):
                    html_chunk += '</html>'
                
                soup = BeautifulSoup(html_chunk, 'html.parser')
                tables = soup.find_all('table')
                
                for table in tables:
                    md_table = html_table_to_markdown(table)
                    final_text = final_text.replace(str(table), md_table, 1)
        
        return final_text
    
    return text

def call_gemini_api(original_text, gemini_key):
    """Gọi Gemini API để sửa chính tả và ngữ pháp tiếng Việt"""
    try:
        if not gemini_key:
            return "Lỗi: Chưa có Gemini API Key"
        
        GEMINI_API_URL = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-flash-002:generateContent?key=" + gemini_key
        )
        
        prompt = (
            "Please help me correct Vietnamese spelling and grammar in the following text. "
            "IMPORTANT: Do not change any image paths, LaTeX formulas, or Vietnamese diacritical marks. "
            "Return only the corrected text with the same structure and markdown formatting:\n\n"
            f"{original_text}"
        )
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192,
            }
        }
        
        headers = {"Content-Type": "application/json"}
        resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=(10, 180))
        
        if resp.status_code == 200:
            data = resp.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    corrected_text = candidate["content"]["parts"][0].get("text", "")
                    if corrected_text.strip():
                        return corrected_text
            
            return "Lỗi: Không thể trích xuất được kết quả từ Gemini API."
        else:
            return f"Lỗi: Gemini API - HTTP {resp.status_code} - {resp.text}"
    
    except Exception as e:
        return f"Lỗi: Gọi Gemini API thất bại: {e}"

def process_pdf(pdf_data, mineru_token, gemini_key, task_id, part_name=None):
    """Xử lý PDF bằng Mineru API và Gemini API"""
    try:
        # Cập nhật trạng thái
        processing_tasks[task_id]["status"] = "uploading"
        processing_tasks[task_id]["progress"] = 5
        
        # Lấy URL upload
        url_batch = "https://mineru.net/api/v4/file-urls/batch"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {mineru_token}"}
        data = {
            "enable_formula": True,
            "enable_table": True,
            "layout_model": "doclayout_yolo",
            "language": "vi",
            "files": [{"name": "demo.pdf", "is_ocr": True, "data_id": "abcd1234"}]
        }
        
        timeout_val = get_timeout(len(pdf_data))
        resp = requests.post(url_batch, headers=headers, json=data, timeout=timeout_val)
        
        if resp.status_code != 200:
            processing_tasks[task_id]["status"] = "error"
            processing_tasks[task_id]["error"] = f"HTTP {resp.status_code}: {resp.text}"
            return
        
        rj = resp.json()
        code = rj.get("code")
        
        if code not in [0, 200]:
            processing_tasks[task_id]["status"] = "error"
            processing_tasks[task_id]["error"] = f"Mã lỗi: {code}, msg: {rj.get('msg')}"
            return
        
        batch_id = rj["data"]["batch_id"]
        file_urls = rj["data"]["file_urls"]
        
        if not file_urls:
            processing_tasks[task_id]["status"] = "error"
            processing_tasks[task_id]["error"] = "Không có link upload trả về."
            return
        
        upload_url = file_urls[0]
        processing_tasks[task_id]["progress"] = 10
        
        # Upload PDF
        up_resp = requests.put(upload_url, data=pdf_data, timeout=timeout_val)
        
        if up_resp.status_code != 200:
            processing_tasks[task_id]["status"] = "error"
            processing_tasks[task_id]["error"] = f"Upload thất bại, HTTP {up_resp.status_code}"
            return
        
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["progress"] = 20
        
        # Poll cho kết quả
        url_get = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        headers_poll = {"Content-Type": "application/json", "Authorization": f"Bearer {mineru_token}"}
        timeout_val_poll = get_timeout(len(pdf_data))
        
        max_retry = 30
        for i in range(max_retry):
            # Kiểm tra nếu người dùng đã hủy task
            if processing_tasks[task_id]["status"] == "cancelled":
                return
                
            time.sleep(5)
            progress = 20 + int(((i+1)/max_retry) * 70)
            processing_tasks[task_id]["progress"] = progress
            
            r = requests.get(url_get, headers=headers_poll, timeout=timeout_val_poll)
            
            if r.status_code == 200:
                rj = r.json()
                code = rj.get("code")
                
                if code in [0, 200]:
                    extract_result = rj["data"].get("extract_result", [])
                    
                    if extract_result:
                        res = extract_result[0]
                        state = res.get("state", "")
                        
                        if state == "done":
                            full_zip_url = res.get("full_zip_url", "")
                            
                            if not full_zip_url:
                                processing_tasks[task_id]["status"] = "error"
                                processing_tasks[task_id]["error"] = "Không tìm thấy link kết quả!"
                                return
                            
                            # Tải và trích xuất markdown
                            processing_tasks[task_id]["status"] = "downloading"
                            md_text = download_and_read_full_md(full_zip_url, task_id)
                            
                            if md_text.startswith("Lỗi:"):
                                processing_tasks[task_id]["status"] = "error"
                                processing_tasks[task_id]["error"] = md_text
                                return
                            
                            # Hiệu đính với Gemini nếu có API key
                            if gemini_key:
                                processing_tasks[task_id]["status"] = "correcting"
                                processing_tasks[task_id]["progress"] = 90
                                corrected_text = call_gemini_api(md_text, gemini_key)
                                
                                if corrected_text.startswith("Lỗi:"):
                                    # Nếu Gemini lỗi, vẫn tiếp tục với kết quả gốc
                                    processing_tasks[task_id]["result"] = md_text
                                    processing_tasks[task_id]["gemini_error"] = corrected_text
                                else:
                                    processing_tasks[task_id]["result"] = corrected_text
                            else:
                                processing_tasks[task_id]["result"] = md_text
                            
                            # Hoàn thành
                            processing_tasks[task_id]["status"] = "completed"
                            processing_tasks[task_id]["progress"] = 100
                            
                            # Thêm tên phần nếu có
                            if part_name:
                                processing_tasks[task_id]["part_name"] = part_name
                                
                            return
                        
                        elif state == "failed":
                            err_msg = res.get("err_msg", "Unknown error")
                            processing_tasks[task_id]["status"] = "error"
                            processing_tasks[task_id]["error"] = f"Task failed: {err_msg}"
                            return
            else:
                processing_tasks[task_id]["status"] = "error"
                processing_tasks[task_id]["error"] = f"Poll thất bại HTTP {r.status_code}"
                return
        
        processing_tasks[task_id]["status"] = "error"
        processing_tasks[task_id]["error"] = "Hết thời gian chờ. Vui lòng thử lại sau."
        return
    
    except Exception as e:
        processing_tasks[task_id]["status"] = "error"
        processing_tasks[task_id]["error"] = f"Lỗi: {str(e)}"
        return

# ------------------------------ Route Decorators ------------------------------
def login_required(f):
    """Decorator yêu cầu đăng nhập"""
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def activation_required(f):
    """Decorator yêu cầu kích hoạt"""
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        
        if 'activation_status' not in session or session['activation_status'] != "ĐÃ KÍCH HOẠT":
            return redirect(url_for('activation_status'))
            
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# ------------------------------ Flask Routes ------------------------------
@app.route('/')
def index():
    """Trang chủ"""
    if 'logged_in' in session and session['logged_in']:
        if session['activation_status'] == "ĐÃ KÍCH HOẠT":
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('activation_status'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Trang đăng nhập"""
    error = None
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if authenticate_user(username, password):
            session['logged_in'] = True
            session['username'] = username
            
            # Tạo và lưu hardware ID
            hardware_id = generate_hardware_id(username)
            session['hardware_id'] = hardware_id
            
            # Kiểm tra trạng thái kích hoạt
            is_activated = check_activation(hardware_id)
            session['activation_status'] = "ĐÃ KÍCH HOẠT" if is_activated else "CHƯA KÍCH HOẠT"
            
            return redirect(url_for('index'))
        else:
            error = 'Tên đăng nhập hoặc mật khẩu không đúng!'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """Đăng xuất"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/activation-status')
@login_required
def activation_status():
    """Trang trạng thái kích hoạt"""
    if 'hardware_id' in session:
        is_activated = check_activation(session['hardware_id'])
        session['activation_status'] = "ĐÃ KÍCH HOẠT" if is_activated else "CHƯA KÍCH HOẠT"
    
    return render_template('activation.html')

@app.route('/dashboard')
@activation_required
def dashboard():
    """Trang chính sau khi đăng nhập và kích hoạt"""
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
@activation_required
def upload_file():
    """API upload file PDF"""
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        # Lưu file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Đọc nội dung file
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Trả về ID file để tiếp tục xử lý
        file_id = hashlib.md5(file_data).hexdigest()
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'size': len(file_data)
        })
    
    return jsonify({'error': 'File không hợp lệ'}), 400

@app.route('/process', methods=['POST'])
@activation_required
def process_uploaded_file():
    """API xử lý file đã upload"""
    data = request.json
    
    if not data or 'file_id' not in data:
        return jsonify({'error': 'Thiếu thông tin file'}), 400
    
    file_id = data['file_id']
    filename = data.get('filename', 'unknown.pdf')
    mineru_token = data.get('mineru_token', '')
    gemini_key = data.get('gemini_key', '')
    
    if not mineru_token:
        return jsonify({'error': 'Thiếu Mineru Token'}), 400
    
    # Tìm file trong thư mục uploads
    file_path = None
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f == secure_filename(filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            break
    
    if not file_path:
        return jsonify({'error': 'Không tìm thấy file đã upload'}), 404
    
    # Đọc file
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    # Tạo task ID
    task_id = f"{session['username']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Khởi tạo thông tin task
    processing_tasks[task_id] = {
        "status": "initialized",
        "progress": 0,
        "filename": filename,
        "started": datetime.now().isoformat(),
        "result": None,
        "error": None
    }
    
    # Bắt đầu xử lý trong thread mới
    thread = threading.Thread(
        target=process_pdf,
        args=(file_data, mineru_token, gemini_key, task_id)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'Đã bắt đầu xử lý'
    })

@app.route('/split-pdf', methods=['POST'])
@activation_required
def split_pdf_file():
    """API tách PDF"""
    data = request.json
    
    if not data or 'file_id' not in data:
        return jsonify({'error': 'Thiếu thông tin file'}), 400
    
    file_id = data['file_id']
    filename = data.get('filename', 'unknown.pdf')
    
    # Tìm file trong thư mục uploads
    file_path = None
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f == secure_filename(filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            break
    
    if not file_path:
        return jsonify({'error': 'Không tìm thấy file đã upload'}), 404
    
    # Đọc file
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    # Tách PDF
    parts = split_pdf(file_data)
    
    # Lưu các phần vào thư mục tạm thời
    split_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"split_{file_id}")
    os.makedirs(split_dir, exist_ok=True)
    
    part_info = []
    for i, part in enumerate(parts):
        part_filename = f"part_{i+1}.pdf"
        part_path = os.path.join(split_dir, part_filename)
        
        with open(part_path, 'wb') as f:
            f.write(part['data'])
        
        part_info.append({
            'id': i+1,
            'name': part['name'],
            'path': part_path,
            'filename': part_filename
        })
    
    # Lưu thông tin các phần vào session
    if 'split_parts' not in session:
        session['split_parts'] = {}
    
    session['split_parts'][file_id] = part_info
    
    return jsonify({
        'success': True,
        'file_id': file_id,
        'parts': [{'id': p['id'], 'name': p['name']} for p in part_info],
        'message': f'Đã tách thành {len(parts)} phần'
    })

@app.route('/process-part', methods=['POST'])
@activation_required
def process_part():
    """API xử lý một phần của PDF đã tách"""
    data = request.json
    
    if not data or 'file_id' not in data or 'part_id' not in data:
        return jsonify({'error': 'Thiếu thông tin file hoặc phần'}), 400
    
    file_id = data['file_id']
    part_id = int(data['part_id'])
    mineru_token = data.get('mineru_token', '')
    gemini_key = data.get('gemini_key', '')
    
    if not mineru_token:
        return jsonify({'error': 'Thiếu Mineru Token'}), 400
    
    # Kiểm tra xem có thông tin về phần không
    if 'split_parts' not in session or file_id not in session['split_parts']:
        return jsonify({'error': 'Không tìm thấy thông tin về phần đã tách'}), 404
    
    # Tìm phần cần xử lý
    part_info = None
    for part in session['split_parts'][file_id]:
        if part['id'] == part_id:
            part_info = part
            break
    
    if not part_info:
        return jsonify({'error': 'Không tìm thấy phần cần xử lý'}), 404
    
    # Đọc file phần
    with open(part_info['path'], 'rb') as f:
        part_data = f.read()
    
    # Tạo task ID
    task_id = f"{session['username']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_part{part_id}"
    
    # Khởi tạo thông tin task
    processing_tasks[task_id] = {
        "status": "initialized",
        "progress": 0,
        "filename": part_info['filename'],
        "started": datetime.now().isoformat(),
        "result": None,
        "error": None
    }
    
    # Bắt đầu xử lý trong thread mới
    thread = threading.Thread(
        target=process_pdf,
        args=(part_data, mineru_token, gemini_key, task_id, part_info['name'])
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': f'Đã bắt đầu xử lý phần {part_info["name"]}'
    })

@app.route('/task-status/<task_id>', methods=['GET'])
@activation_required
def task_status(task_id):
    """API kiểm tra trạng thái task"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Không tìm thấy task'}), 404
    
    task_info = processing_tasks[task_id].copy()
    
    # Không trả về kết quả đầy đủ nếu task chưa hoàn thành
    if task_info['status'] != 'completed':
        if 'result' in task_info:
            del task_info['result']
    
    return jsonify(task_info)

@app.route('/task-result/<task_id>', methods=['GET'])
@activation_required
def task_result(task_id):
    """API lấy kết quả task"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Không tìm thấy task'}), 404
    
    task_info = processing_tasks[task_id]
    
    if task_info['status'] != 'completed':
        return jsonify({'error': 'Task chưa hoàn thành'}), 400
    
    return jsonify({
        'success': True,
        'result': task_info['result'],
        'part_name': task_info.get('part_name')
    })

@app.route('/cancel-task/<task_id>', methods=['POST'])
@activation_required
def cancel_task(task_id):
    """API hủy task đang xử lý"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Không tìm thấy task'}), 404
    
    task_info = processing_tasks[task_id]
    
    if task_info['status'] in ['completed', 'error']:
        return jsonify({'error': 'Task đã kết thúc, không thể hủy'}), 400
    
    # Đánh dấu task là đã hủy
    processing_tasks[task_id]['status'] = 'cancelled'
    
    return jsonify({
        'success': True,
        'message': 'Đã hủy task'
    })

@app.route('/images/<task_id>/<image_name>')
@activation_required
def serve_image(task_id, image_name):
    """Phục vụ hình ảnh từ thư mục output"""
    image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"task_{task_id}", "images", image_name)
    
    if not os.path.exists(image_path):
        return "Image not found", 404
    
    return send_file(image_path)

@app.route('/download/<task_id>/<format>')
@activation_required
def download_result(task_id, format):
    """Tải xuống kết quả dưới dạng markdown hoặc text"""
    if task_id not in processing_tasks:
        return "Task not found", 404
    
    task_info = processing_tasks[task_id]
    
    if task_info['status'] != 'completed' or not task_info['result']:
        return "Result not available", 400
    
    result_text = task_info['result']
    
    if format == 'text':
        # Chuyển đổi bảng trong markdown
        result_text = convert_tables_to_md(result_text)
        
        return send_file(
            io.BytesIO(result_text.encode('utf-8')),
            mimetype='text/plain',
            as_attachment=True,
            download_name=f"result_{task_id}.txt"
        )
    
    elif format == 'markdown':
        return send_file(
            io.BytesIO(result_text.encode('utf-8')),
            mimetype='text/markdown',
            as_attachment=True,
            download_name=f"result_{task_id}.md"
        )
    
    else:
        return "Invalid format", 400

# ------------------------------ Main ------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
