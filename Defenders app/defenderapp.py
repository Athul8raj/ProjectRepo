from flask import Flask, render_template, request,redirect, url_for, send_from_directory
import requests,string,time,urllib,os
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/IsItPlagiarized/', methods=['POST'])
def IsItPlagiarized():
    text_to_filter = request.form['text_to_check']
    if (text_to_filter.lstrip().rstrip() == ''):
        return render_template('index.html')
    punkt_param = PunktParameters()
    sentence_splitter = PunktSentenceTokenizer(punkt_param)
    sentences = sentence_splitter.tokenize(text_to_filter)
    probability_of_plagiarism = 0
    for a_sentence in sentences:
        time.sleep(0.3)
        content = str(filter(lambda x: x in string.printable, a_sentence))
        the_term = urllib.parse.quote('+' + '"' + content + '"')
        page = requests.get('https://www.bing.com/search?q=' + the_term)
        if not "No results found for" in page.text:
            probability_of_plagiarism += 1
    is_it_plagiarized = str((probability_of_plagiarism / len(sentences)) * 100) + '%'
    return render_template('result.html', text_to_filter=text_to_filter, is_it_plagiarized=is_it_plagiarized)

dir_name = os.path.dirname(__file__)
#app.url_map.converters['list'] = ListConverter
UPLOAD_FOLDER = os.path.join(dir_name)
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'csv', 'xls', 'xlsx', 'doc', 'docx'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['APP_SECRET_KEY'] = 'my_secret_key'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_file/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('convert_dtypes',filename=filename))
    return render_template('memory_usage.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename,as_attachment=True)


@app.route('/convert_dtypes/<filename>') 
def convert_dtypes(filename): 
    if '.csv' in filename.lower():
        g1 = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    else:
        g1 = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    usage_before = round((g1.memory_usage(deep=True).sum())/1024**2,3)
    for dtype in ['float64', 'int64', 'object']:
        selected_dtype = g1.select_dtypes(include=[dtype])
        if dtype == 'int64':
            converted_int = selected_dtype.apply(pd.to_numeric, downcast='unsigned')
        if dtype == 'float64':
            converted_float = selected_dtype.apply(pd.to_numeric, downcast='float')
        if dtype == 'object':
            converted_obj = pd.DataFrame()
            for col in selected_dtype.columns:
                num_unique_values = len(selected_dtype[col].unique())
                num_total_values = len(selected_dtype[col])
                if num_unique_values / num_total_values < 0.5:
                    converted_obj.loc[:, col] = selected_dtype[col].astype('category')
                else:
                    converted_obj.loc[:, col] = selected_dtype[col]
            converted_dtype = [converted_int, converted_float, converted_obj]
            for conv_dtype in converted_dtype:
                if isinstance(conv_dtype, pd.DataFrame):
                    usage_b = conv_dtype.memory_usage(deep=True).sum()
                else:
                    usage_b = conv_dtype.memory_usage(deep=True)
                usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
                usage_mb = round(usage_mb,3)
                percent_diff = round(((usage_before - usage_mb)/usage_before)*100,2)
            return render_template('mem_report.html', usage_mb=usage_mb,usage_before=usage_before,percent_diff=percent_diff)


if __name__ == '__main__':
    app.run(debug=True)
