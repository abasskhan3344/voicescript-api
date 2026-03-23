from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, time
from collections import defaultdict

app = Flask(__name__)
CORS(app)

GROQ_KEY = os.environ.get('GROQ_KEY', '')
OPENROUTER_KEY = os.environ.get('OPENROUTER_KEY', '')

request_counts = defaultdict(list)

def rate_ok(ip, mx=10, win=3600):
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now-t < win]
    if len(request_counts[ip]) >= mx:
        return False
    request_counts[ip].append(now)
    return True

@app.route('/')
def home():
    return jsonify({'status': 'ok', 'app': 'VoiceScript AI'})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not rate_ok(request.remote_addr):
        return jsonify({'error': 'Rate limit. Try in 1 hour.'}), 429
    if not GROQ_KEY:
        return jsonify({'error': 'Server not configured'}), 500
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    f = request.files['audio']
    lang = request.form.get('language', '')
    mode = request.form.get('mode', 'accurate')
    models = {'fast':'whisper-large-v3-turbo','balanced':'whisper-large-v3-turbo','accurate':'whisper-large-v3'}
    
    f.seek(0,2); sz = f.tell(); f.seek(0)
    if sz > 25*1024*1024:
        return jsonify({'error': 'File too large. Max 25MB.'}), 400

    try:
        audio_data = f.read()
        files = {'file': (f.filename or 'audio.mp3', audio_data, 'audio/mpeg')}
        data = {'model': models.get(mode,'whisper-large-v3'), 'response_format': 'verbose_json'}
        if lang:
            data['language'] = lang
            prompts = {'ps':'دا د پښتو ژبې غږیز لیکنه ده.','ur':'یہ اردو زبان میں گفتگو ہے۔','ar':'هذا تسجيل باللغة العربية.'}
            if lang in prompts: data['prompt'] = prompts[lang]

        r = requests.post('https://api.groq.com/openai/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {GROQ_KEY}'}, files=files, data=data, timeout=60)

        if not r.ok:
            return jsonify({'error': r.json().get('error',{}).get('message','Failed')}), r.status_code

        res = r.json()
        tx = res.get('text','').strip()
        dl = res.get('language','unknown').lower()
        lm = {'urdu':'ur','ur':'ur','pashto':'ps','pushto':'ps','ps':'ps','english':'en','en':'en',
              'arabic':'ar','ar':'ar','persian':'fa','fa':'fa','hindi':'hi','hi':'hi'}
        dl = lm.get(dl, dl)
        if lang: dl = lang
        return jsonify({'transcript': tx, 'language': dl, 'words': len(tx.split())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    if not rate_ok(request.remote_addr, mx=20):
        return jsonify({'error': 'Rate limit. Try later.'}), 429
    if not OPENROUTER_KEY:
        return jsonify({'error': 'Server not configured'}), 500

    d = request.json or {}
    tx = d.get('text','')
    src = d.get('source','ps')
    tgt = d.get('target','ur')
    if not tx: return jsonify({'error': 'No text'}), 400

    names = {'ur':'Urdu','ps':'Pashto','en':'English','ar':'Arabic','fa':'Persian','hi':'Hindi',
             'fr':'French','de':'German','es':'Spanish','zh':'Chinese','tr':'Turkish','ru':'Russian',
             'ja':'Japanese','ko':'Korean','it':'Italian','pt':'Portuguese'}

    if src=='ps' and tgt=='ur':
        sys = 'Aap ek maahir Pashto se Urdu translator hain. Pashto matn ko KHALIS Urdu mein translate karein. Sirf Urdu likhein. Koi Pashto lafz na aaye. Sirf translation return karein.'
    elif src=='ps' and tgt=='en':
        sys = 'Expert Pashto to English translator. Return ONLY English translation.'
    else:
        sys = f'Expert translator. Translate {names.get(src,src)} to {names.get(tgt,tgt)}. Return ONLY the translation.'

    try:
        r = requests.post('https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization':f'Bearer {OPENROUTER_KEY}','Content-Type':'application/json',
                     'HTTP-Referer':'https://voicescript-ai.netlify.app','X-Title':'VoiceScript AI'},
            json={'model':'google/gemini-2.0-flash-001','messages':[{'role':'system','content':sys},{'role':'user','content':tx}],
                  'max_tokens':3000,'temperature':0.1}, timeout=30)

        if r.ok:
            return jsonify({'translation': r.json()['choices'][0]['message']['content'].strip(), 'service':'Gemini'})

        # Fallback Groq
        r2 = requests.post('https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization':f'Bearer {GROQ_KEY}','Content-Type':'application/json'},
            json={'model':'llama-3.3-70b-versatile','messages':[{'role':'system','content':sys},{'role':'user','content':tx}],
                  'max_tokens':2000,'temperature':0.1}, timeout=30)
        if r2.ok:
            return jsonify({'translation': r2.json()['choices'][0]['message']['content'].strip(), 'service':'Groq'})
        return jsonify({'error': 'Translation failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=False)
