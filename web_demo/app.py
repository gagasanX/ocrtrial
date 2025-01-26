from flask import Flask, request, jsonify, render_template_string
from paddleocr import PaddleOCR
import os
import anthropic
from dotenv import load_dotenv
from datetime import datetime
import logging
import cv2
import numpy as np
from PIL import Image
import io
from threading import Timer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
app.config['TIMEOUT_DURATION'] = 60  # 60 seconds timeout

# Initialize PaddleOCR with optimized settings
ocr = PaddleOCR(
    use_angle_cls=True,  # Change back to True
    lang='en',
    use_gpu=False,
    enable_mkldnn=True,
    cpu_threads=4
)

# Setup Anthropic client
client = anthropic.Anthropic(
    api_key=os.getenv('ANTHROPIC_API_KEY')
)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI-Powered OCR System</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <style>
        .loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 999;
            justify-content: center;
            align-items: center;
            display: none;
        }
        .loading-content {
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            max-width: 80%;
            width: 400px;
        }
        .processing-step {
            margin-top: 10px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="loading" id="loadingOverlay">
        <div class="loading-content">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p id="processingStatus" class="font-semibold">Processing your document...</p>
            <p id="processingStep" class="processing-step">Initializing...</p>
        </div>
    </div>

    <div class="container mx-auto px-4 py-8">
        <div class="max-w-2xl mx-auto">
            <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
                <h1 class="text-2xl font-bold mb-4 text-gray-800">Document Amanie Scanner</h1>
                <p class="text-gray-600 mb-4">Upload your document for 13-step AI verification</p>
                
                <form id="uploadForm" action="/ocr" method="post" enctype="multipart/form-data" class="space-y-4">
                    <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                        <input type="file" 
                               name="image" 
                               accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff"
                               class="w-full" 
                               required
                               onchange="validateFile(this)">
                        <p class="text-sm text-gray-500 mt-2">Supported formats: PDF, JPEG, PNG, BMP, TIFF</p>
                        <p class="text-sm text-gray-500">Maximum file size: 10MB</p>
                    </div>
                    <button type="submit" 
                            class="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-300">
                        Process Document
                    </button>
                </form>
            </div>
            
            <div id="results" class="bg-white rounded-lg shadow-lg p-6 hidden">
                <h2 class="text-xl font-semibold mb-4">Results</h2>
                <div id="resultContent" class="space-y-4"></div>
            </div>
        </div>
    </div>

    <script>
        function updateProcessingStep(step) {
            document.getElementById('processingStep').textContent = step;
        }

        function validateFile(input) {
            const file = input.files[0];
            if (!file) return false;

            const validTypes = [
                'application/pdf',
                'image/jpeg',
                'image/png',
                'image/bmp',
                'image/tiff'
            ];
            
            const maxSize = 10 * 1024 * 1024; // 10MB
            
            if (!validTypes.includes(file.type)) {
                Swal.fire('Error', 'Please upload a valid document (PDF, JPEG, PNG, BMP, TIFF)', 'error');
                input.value = '';
                return false;
            }
            
            if (file.size > maxSize) {
                Swal.fire('Error', 'File size must be less than 10MB', 'error');
                input.value = '';
                return false;
            }
            
            return true;
        }

        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            
            const form = e.target;
            const file = form.querySelector('input[type="file"]').files[0];
            
            if (!file || !validateFile(form.querySelector('input[type="file"]'))) {
                return;
            }
            
            const loadingOverlay = document.getElementById('loadingOverlay');
            loadingOverlay.style.display = 'flex';
            
            const formData = new FormData(form);
            
            try {
                updateProcessingStep('Processing document...');
                
                const response = await fetch('/ocr', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Server error');
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                document.getElementById('results').classList.remove('hidden');
                document.getElementById('resultContent').innerHTML = `
                    <div class="p-4 bg-green-50 rounded-lg">
                        <h3 class="font-semibold text-green-800">Extracted Text</h3>
                        <pre class="mt-2 text-sm text-gray-600 whitespace-pre-wrap">${JSON.stringify(data.text, null, 2)}</pre>
                    </div>
                    <div class="p-4 bg-blue-50 rounded-lg">
                        <h3 class="font-semibold text-blue-800">Validation Results</h3>
                        <pre class="mt-2 text-sm text-gray-600 whitespace-pre-wrap">${JSON.stringify(data.validation, null, 2)}</pre>
                    </div>
                `;
            } catch (error) {
                Swal.fire('Error', error.message, 'error');
            } finally {
                loadingOverlay.style.display = 'none';
            }
        };
    </script>
</body>
</html>
'''

def process_with_timeout(func, timeout_duration):
    """Handle timeouts without using signals"""
    result = None
    is_timeout = False
    
    def handle_timeout():
        nonlocal is_timeout
        is_timeout = True
    
    timer = Timer(timeout_duration, handle_timeout)
    try:
        timer.start()
        result = func()
    finally:
        timer.cancel()
    
    if is_timeout:
        raise TimeoutError("Processing timed out")
    return result

def preprocess_image(file_data, max_size=1800):
    """Handle both images and PDFs with optimized preprocessing"""
    try:
        # Handle PDF
        if hasattr(file_data, 'filename') and file_data.filename.lower().endswith('.pdf'):
            import fitz  # PyMuPDF
            
            # Read PDF content
            file_content = file_data.read()
            file_data.seek(0)  # Reset file pointer
            
            # Open PDF
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            first_page = pdf_document[0]
            
            # Convert to image with higher resolution
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            pix = first_page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf_document.close()
        else:
            # Handle regular image
            image = Image.open(file_data)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple([int(dim * ratio) for dim in image.size])
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Convert back to PIL and compress
        processed_image = Image.fromarray(denoised)
        output = io.BytesIO()
        processed_image.save(output, format='JPEG', quality=85, optimize=True)
        output.seek(0)
        
        return Image.open(output)
    
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise

def validate_with_haiku(text_results):
    """Validate OCR results using Claude Haiku"""
    try:
        prompt = f"""Please validate the following OCR results with 13-step verification:
        {text_results}
        
        Perform these checks in sequence:
        1. Character-level verification (check each character for accuracy)
        2. Number sequence validation (verify numerical sequences)
        3. Format pattern matching (validate document structure)
        4. Checksum verification (verify numerical consistency)
        5. Context-based validation (ensure logical content)
        6. Cross-reference check (verify related fields)
        7. Pattern consistency (check formatting consistency)
        8. Special character validation (verify symbols and special chars)
        9. Field boundary verification (check field separations)
        10. Data type conformity (validate data types)
        11. Range validation (verify value ranges)
        12. Historical comparison (compare with typical patterns)
        13. Final integrity check (overall validation)
        
        Return a JSON object with confidence scores and validation status.
        """

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {"error": "Validation failed", "details": str(e)}

@app.route('/')
def home():
    """Render the home page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/ocr', methods=['POST'])
def process_image():
    """Process uploaded image/PDF and run OCR"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Validate file type
        allowed_extensions = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type'})

        # Create uploads directory
        os.makedirs('uploads', exist_ok=True)
        
        def process():
            # Preprocess image
            processed_image = preprocess_image(file)
            
            # Save processed file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join('uploads', f'{timestamp}_processed.jpg')
            processed_image.save(file_path, 'JPEG', quality=85, optimize=True)
            
            try:
                # Run OCR with modified parameters
                result = ocr.ocr(file_path) 
                
                # Process results - Improved extraction
                text_results = []
                if result:
                    for idx in range(len(result)):
                        if result[idx]:
                            for line in result[idx]:
                                if len(line) >= 2 and line[1]:
                                    text_results.append(line[1][0])  # Extract text
                
                return text_results, timestamp
                
            finally:
                # Cleanup
                if os.path.exists(file_path):
                    os.remove(file_path)

        # Run processing with timeout
        text_results, timestamp = process_with_timeout(process, app.config['TIMEOUT_DURATION'])
        
        # Validate results
        if text_results:
            validation_results = validate_with_haiku(text_results)
        else:
            validation_results = {
                "error": "No text detected",
                "suggestion": "Please try with a clearer document"
            }
        
        return jsonify({
            'text': text_results,
            'validation': validation_results,
            'timestamp': timestamp,
            'file_type': file.filename.rsplit('.', 1)[1].lower(),
            'processing_completed': True
        })

    except TimeoutError:
        return jsonify({
            'error': 'Processing timeout. Please try with a clearer document or smaller file.'
        })
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({
            'error': str(e),
            'suggestion': 'Please try again with a different document'
        })

if __name__ == '__main__':
    # Initialize upload directory
    os.makedirs('uploads', exist_ok=True)
    
    # Clean existing temporary files
    for file in os.listdir('uploads'):
        try:
            os.remove(os.path.join('uploads', file))
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)