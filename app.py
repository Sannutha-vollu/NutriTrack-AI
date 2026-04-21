from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import score_foods

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'NutriTrack AI Engine Running [OK]'})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    profile = data.get('profile', {})
    foods = data.get('foods', [])
    context = data.get('context', {})

    if not foods:
        return jsonify({'error': 'No foods provided'}), 400

    results = score_foods(foods, profile, context)
    return jsonify(results)

if __name__ == '__main__':
    print("[INFO] NutriTrack AI Engine starting on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)
