from app import app
print('🚀 Starting AI Medical Image Classifier Flask Server...')
print('📱 Open your browser and visit: http://localhost:5000')
print('📚 Documentation page: http://localhost:5000/documentation')
print('❌ Press Ctrl+C to stop the server')
print('=' * 60)
app.run(host='localhost', port=5000, debug=False)