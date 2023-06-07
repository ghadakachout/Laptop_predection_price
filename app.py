from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# Charger le modèle et les données
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict_price():
    if request.method == 'POST':

        # Obtenez les données du formulaire
        company = request.form['company']
        laptop_type = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = request.form['touchscreen']
        ips = request.form['ips']
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']

        # Préparez les données pour la prédiction
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        # Effectuez la prédiction
        query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)
        predicted_price = int(np.exp(pipe.predict(query)[0]))

        return render_template('resultat.html', output=predicted_price)

    else:
        # Renvoyer le formulaire de saisie des données
        brands = df['Company'].unique()
        types = df['TypeName'].unique()
        ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
        resolutions = ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
        hdd_options = [0, 128, 256, 512, 1024, 2048]
        ssd_options = [0, 8, 128, 256, 512, 1024]
        cpu_brands = df['Cpu brand'].unique()
        gpu_brands = df['Gpu brand'].unique()
        os_options = df['os'].unique()

        return render_template('index.html', brands=brands, types=types, ram_options=ram_options, resolutions=resolutions,
                               hdd_options=hdd_options, ssd_options=ssd_options, cpu_brands=cpu_brands, gpu_brands=gpu_brands,
                               os_options=os_options)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
