import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
import pyqtgraph as pg
from PyQt5.QtGui import QFont
from scipy.interpolate import make_interp_spline, interp1d
from scipy.signal import argrelextrema
from scipy.optimize import differential_evolution
from stmf_processor import stmf_to_dataframe
from TireAnalysis import TireAnalysis
import re



class TireProcessingPage(QWidget):
    def __init__(self, tire_path, parent):
        super().__init__()
        self.tire_path = tire_path
        self.parent = parent
        self.datasets = {}
        self.results = {}
        self.initUI()
        self.process_all_conditions()

    def initUI(self):
        self.setWindowTitle('Processing Tire Data')
        self.setStyleSheet("background-color: #181c36; color: white;")

        layout = QVBoxLayout(self)

        self.plot_widget = pg.PlotWidget(title='Courbes lissées de P_Brake_Hydraulic', background='w')
        layout.addWidget(self.plot_widget)

        self.results_widget = QTextEdit(self)
        self.results_widget.setReadOnly(True)
        layout.addWidget(self.results_widget)

        button_layout = QHBoxLayout()

        self.optimize_button = QPushButton('Optimize Curve')
        self.optimize_button.clicked.connect(self.on_optimize_button_clicked)
        button_layout.addWidget(self.optimize_button)

        # Ajouter les boutons pour chaque condition et sous-condition
        self.condition_buttons = {
            ('FZ40', 'ND'): QPushButton('FZ40 ND'),
            ('FZ40', 'NW'): QPushButton('FZ40 NW'),
            ('FZ80', 'ND'): QPushButton('FZ80 ND'),
            ('FZ80', 'NW'): QPushButton('FZ80 NW'),
            ('FZ120', 'ND'): QPushButton('FZ120 ND'),
            ('FZ120', 'NW'): QPushButton('FZ120 NW')
        }
        for (condition, subcondition), button in self.condition_buttons.items():
            button.clicked.connect(lambda checked, c=condition, s=subcondition: self.process_and_plot_data(c, s))
            button_layout.addWidget(button)

        back_button = QPushButton('Back to Tire Selection')
        back_button.clicked.connect(self.goBack)
        button_layout.addWidget(back_button)

        layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Accéder aux axes X et Y
        axisX = self.plot_widget.getAxis('bottom')
        axisY = self.plot_widget.getAxis('left')

        # Changer la couleur de la ligne des axes en blanc
        axisX.setPen('black')
        axisY.setPen('black')

        # Configurer la police des tick labels
        axisX.setStyle(tickFont=QFont('Arial', 10), tickTextOffset=10)
        axisY.setStyle(tickFont=QFont('Arial', 10), tickTextOffset=10)


    def on_optimize_button_clicked(self):
        self.results_widget.clear()
        self.process_all_conditions()
        self.plot_all_results()
     
    


    def process_all_conditions(self):
        conditions = ["FZ40", "FZ80", "FZ120"]
        subconditions = ["ND", "NW"]

        for condition in conditions:
            for subcondition in subconditions:
                subcondition_path = os.path.join(self.tire_path, condition, subcondition)
                if os.path.exists(subcondition_path):
                    self.processData(subcondition_path, condition, subcondition)
                    self.optimize_and_store_results(condition, subcondition)
                else:
                    print(f"Le chemin {subcondition_path} n'existe pas.")  # Debugging line
                self.print_results(condition, subcondition)

        
        self.launch_tire_analysis()
            
       
       

    def processData(self, path, condition, subcondition):
        file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".stmf")]

        datasets = []
        for path in file_paths:
            dataset = stmf_to_dataframe(path)
            datasets.append(dataset)

        if len(datasets) >= 3:
            self.data1 = datasets[0][['P_Brake_Hydraulic', 'Fx_MM', 'Fz_MM', 'SRNew']]
            self.data2 = datasets[1][['P_Brake_Hydraulic', 'Fx_MM', 'Fz_MM', 'SRNew']]
            self.data3 = datasets[2][['P_Brake_Hydraulic', 'Fx_MM', 'Fz_MM', 'SRNew']]

            # orrection du bug ici "AttributeError: 'TireProcessingPage' object has no attribute 'min_index1'" 
            self.min_index1, self.max_index1 = self.plot_smoothed_data_with_markers_helper(self.data1)
            self.min_index2, self.max_index2 = self.plot_smoothed_data_with_markers_helper(self.data2)
            self.min_index3, self.max_index3 = self.plot_smoothed_data_with_markers_helper(self.data3)

            self.filter_and_merge_datasets(condition, subcondition)
        else:
            print(f"Not enough data files to process for {condition} {subcondition}")  # Debugging line


    def plot_smoothed_data_with_markers_helper(self, df):
        if df['P_Brake_Hydraulic'].dtype == 'object':
            df['P_Brake_Hydraulic'] = df['P_Brake_Hydraulic'].str.replace(',', '.').astype(float)

        p_brake_hydraulic = df['P_Brake_Hydraulic']
        x = np.arange(len(p_brake_hydraulic))

        x_smooth = np.linspace(x.min(), x.max(), 500)
        spl = make_interp_spline(x, p_brake_hydraulic, k=3)
        y_smooth = spl(x_smooth)

        main_peak_index = np.argmax(y_smooth)
        peak_indices = argrelextrema(y_smooth[:main_peak_index + 1], np.greater)[0]
        min_indices = argrelextrema(y_smooth[:main_peak_index + 1], np.less)[0]

        if peak_indices.size > 0:
            highest_pre_peak_index = peak_indices[-1]
        else:
            highest_pre_peak_index = main_peak_index

        min_before_highest_peak_index = [idx for idx in min_indices if idx < highest_pre_peak_index]

        if min_before_highest_peak_index:
            final_min_index = min_before_highest_peak_index[-1]
        else:
            final_min_index = -1

        interp_func = interp1d(y_smooth, x_smooth)
        df_main_peak_index = interp_func(y_smooth[main_peak_index])
        df_final_min_index = -1
        if final_min_index != -1:
            df_final_min_index = interp_func(y_smooth[final_min_index])

        df_main_peak_index = np.round(df_main_peak_index).astype(int)
        df_final_min_index = np.round(df_final_min_index).astype(int) if df_final_min_index != -1 else -1

        df_main_peak_index = np.clip(df_main_peak_index, 0, len(df) - 1)
        df_final_min_index = np.clip(df_final_min_index, 0, len(df) - 1) if df_final_min_index != -1 else -1

        return df_final_min_index, df_main_peak_index


    def filter_and_merge_datasets(self, condition, subcondition):
        dataset1 = self.data1.iloc[self.min_index1:self.max_index1]
        dataset2 = self.data2.iloc[self.min_index2:self.max_index2]
        dataset3 = self.data3.iloc[self.min_index3:self.max_index3]

        final_dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)

        final_dataset['Fx_MM'] = pd.to_numeric(final_dataset['Fx_MM'], errors='coerce')
        final_dataset['Fz_MM'] = pd.to_numeric(final_dataset['Fz_MM'], errors='coerce')
        final_dataset['SRNew'] = pd.to_numeric(final_dataset['SRNew'], errors='coerce')

        final_dataset['Mux'] = final_dataset['Fx_MM'] / final_dataset['Fz_MM']

        self.datasets[(condition, subcondition)] = final_dataset


    def optimize_curve(self, SR, Mux, normalize=True):
        def estimate_initial_params(SR, Mux):
            D_initial = max(Mux)
            C_initial = 1.0
            B_initial = 1.0 / (max(SR) - min(SR))
            E_initial = 0.0
            return B_initial, C_initial, D_initial, E_initial

        def model_func(SR, B, C, D, E):
            return D * np.sin(C * np.arctan(B * SR - E * (B * SR - np.arctan(B * SR))))

        if normalize:
            SR_max = max(abs(SR))
            Mux_max = max(abs(Mux))
            SR_norm = SR / SR_max
            Mux_norm = Mux / Mux_max
        else:
            SR_norm = SR
            Mux_norm = Mux
            SR_max = 1
            Mux_max = 1

        bounds = [(-5.0, 5.0), (0.1, 10.0), (0.1, 10.0), (0.1, 5.0)]
        initial_params = estimate_initial_params(SR_norm, Mux_norm)

        result = differential_evolution(lambda params: np.sqrt(np.mean((model_func(SR_norm, *params) - Mux_norm)**2)),
                                        bounds, maxiter=5000, popsize=20)  # Supprimer init=initial_params
        B_opt, C_opt, D_opt, E_opt = result.x

        #print('Coefficient B optimal :', B_opt)
        #print('Coefficient C optimal :', C_opt)
        #print('Coefficient D optimal :', D_opt)
        #print('Coefficient E optimal :', E_opt)

        Mux_pred_norm = model_func(SR_norm, B_opt, C_opt, D_opt, E_opt)
        error = np.sqrt(np.mean((Mux_pred_norm - Mux_norm)**2))
        #print('Taux d\'erreur :', error)

        SR_opt_norm = np.linspace(min(SR_norm), max(SR_norm), 500)
        Mux_opt_norm = model_func(SR_opt_norm, B_opt, C_opt, D_opt, E_opt)

        SR_opt = SR_opt_norm * SR_max
        Mux_opt = Mux_opt_norm * Mux_max

        idx_max = np.argmin(Mux_opt)

        Mux_max_val = Mux_opt[idx_max]
        SR_optimal = SR_opt[idx_max]

        #print('Mux_max :', Mux_max_val)
        #print('Slip Ratio optimal :', SR_optimal)

        return SR_opt, Mux_opt, B_opt, C_opt, D_opt, E_opt, error, Mux_max_val, SR_optimal

    def optimize_and_store_results(self, condition, subcondition):
        dataset = self.datasets[(condition, subcondition)]
        SR = dataset['SRNew'].values
        Mux = dataset['Mux'].values

        # Optimiser les courbes et obtenir les résultats
        SR_opt, Mux_opt, B_opt, C_opt, D_opt, E_opt, error, mux_max, sr_opt = self.optimize_curve(SR, Mux)

        # Stocker les valeurs de mu_max et optimal_slip_ratio pour chaque sous-condition
        if subcondition == 'ND':
            self.mu_max_dry = mux_max
            self.optimal_slip_ratio_dry = sr_opt
            self.results[(condition, 'ND')] = {
                'SR_opt': SR_opt,
                'Mux_opt': Mux_opt,
                'B': B_opt,
                'C': C_opt,
                'D': D_opt,
                'E': E_opt,
                'error': error,
                'mux_max': mux_max,
                'sr_opt': sr_opt,
                'BrakingStiffnessDry': (mux_max - 0) / sr_opt,
                'BrakingStiffnessWet': None,
                'MuxWetDry': None,
                'SRWetDry': None
            }
        elif subcondition == 'NW':
            self.mu_max_wet = mux_max
            self.optimal_slip_ratio_wet = sr_opt
            self.results[(condition, 'NW')] = {
                'SR_opt': SR_opt,
                'Mux_opt': Mux_opt,
                'B': B_opt,
                'C': C_opt,
                'D': D_opt,
                'E': E_opt,
                'error': error,
                'mux_max': mux_max,
                'sr_opt': sr_opt,
                'BrakingStiffnessDry':None,
                'BrakingStiffnessWet': (mux_max - 0) / sr_opt,
                'MuxWetDry': None,
                'SRWetDry': None
            }

        # Calculer MuxWetDry et SRWetDry après avoir stocké les résultats des deux sous-conditions
        if (condition, 'ND') in self.results and (condition, 'NW') in self.results:
            mu_max_dry = self.results[(condition, 'ND')]['mux_max']
            mu_max_wet = self.results[(condition, 'NW')]['mux_max']
            optimal_slip_ratio_dry = self.results[(condition, 'ND')]['sr_opt']
            optimal_slip_ratio_wet = self.results[(condition, 'NW')]['sr_opt']

            MuxWetDry = ((mu_max_dry - mu_max_wet) / mu_max_dry * 100) if mu_max_dry and mu_max_wet else None
            SRWetDry = ((optimal_slip_ratio_dry - optimal_slip_ratio_wet) / optimal_slip_ratio_dry * 100) if optimal_slip_ratio_dry and optimal_slip_ratio_wet else None

            self.results[(condition, 'ND')]['MuxWetDry'] = MuxWetDry
            self.results[(condition, 'NW')]['MuxWetDry'] = MuxWetDry
            self.results[(condition, 'ND')]['SRWetDry'] = SRWetDry
            self.results[(condition, 'NW')]['SRWetDry'] = SRWetDry

            # Mettre à jour BrakingStiffnessWet dans ND
            self.results[(condition, 'ND')]['BrakingStiffnessWet'] = self.results[(condition, 'NW')]['BrakingStiffnessWet']

            # Mettre à jour BrakingStiffnessDry dans NW
            self.results[(condition, 'NW')]['BrakingStiffnessDry'] = self.results[(condition, 'ND')]['BrakingStiffnessDry']
       
        


    def process_and_plot_data(self, condition, subcondition):
        if (condition, subcondition) in self.datasets:
            self.plot_widget.clear()
            self.plot_widget.setBackground('white')

            dataset = self.datasets[(condition, subcondition)]
            SR = dataset['SRNew'].values
            Mux = dataset['Mux'].values

            result = self.results.get((condition, subcondition), {})
            SR_opt = result.get('SR_opt', [])
            Mux_opt = result.get('Mux_opt', [])

            self.plot_widget.plot(SR, Mux, pen=None, symbol='o', symbolSize=5, symbolBrush='b', name=f'Data ({condition}, {subcondition})')
            self.plot_widget.plot(SR_opt, Mux_opt, pen=pg.mkPen('r', width=2), name=f'Optimized Curve ({condition}, {subcondition})')

            self.results_widget.clear()
            self.results_widget.append(f"Condition: {condition}, Subcondition: {subcondition}")
            if (condition, subcondition) in self.results:
                res = self.results[(condition, subcondition)]
                self.results_widget.append(f"Taux d'erreur de l'optimisation : {res['error']}")
                self.results_widget.append(f"La valeur du Mux_Max : {res['mux_max']}")
                self.results_widget.append(f"La valeur du SR Optimal : {res['sr_opt']}")
                self.results_widget.append("La valeur des coefficients Optimale après Optimisation : B = {:.3f}, C = {:.3f}, D = {:.3f}, E = {:.3f}".format(res['B'], res['C'], res['D'], res['E']))

                # Afficher les nouvelles valeurs en vérifiant les None
                self.results_widget.append(f"Braking Stiffness Dry : {res['BrakingStiffnessDry']:.3f}" if res.get('BrakingStiffnessDry') is not None else "Braking Stiffness Dry : Non disponible")
                self.results_widget.append(f"Braking Stiffness Wet : {res['BrakingStiffnessWet']:.3f}" if res.get('BrakingStiffnessWet') is not None else "Braking Stiffness Wet : Non disponible")
                self.results_widget.append(f"Mux Wet/Dry : {res['MuxWetDry']:.3f}" if res.get('MuxWetDry') is not None else "Mux Wet/Dry : Non disponible")
                self.results_widget.append(f"SR Wet/Dry : {res['SRWetDry']:.3f}" if res.get('SRWetDry') is not None else "SR Wet/Dry : Non disponible")

                self.results_widget.append("\n")
              
        else:
            self.results_widget.setText(f"Aucune donnée disponible pour {condition} {subcondition}")

    # Debgug console 
    def print_results(self,condition, subcondition):
         if (condition, subcondition) in self.results:
            res = self.results[(condition, subcondition)]
            print(f"Condition: {condition}, Subcondition: {subcondition}")
            print(f"Taux d'erreur de l'optimisation : {res['error']}")
            print(f"La valeur du Mux_Max : {res['mux_max']}")
            print(f"La valeur du SR Optimal : {res['sr_opt']}")
            print("La valeur des coefficients Optimale après Optimisation : B = {:.3f}, C = {:.3f}, D = {:.3f}, E = {:.3f}".format(res['B'], res['C'], res['D'], res['E']))

            # Afficher les nouvelles valeurs en vérifiant les None
            print(f"Braking Stiffness Dry : {res['BrakingStiffnessDry']:.3f}" if res.get('BrakingStiffnessDry') is not None else "Braking Stiffness Dry : Non disponible")
            print(f"Braking Stiffness Wet : {res['BrakingStiffnessWet']:.3f}" if res.get('BrakingStiffnessWet') is not None else "Braking Stiffness Wet : Non disponible")
            print(f"Mux Wet/Dry : {res['MuxWetDry']:.3f}" if res.get('MuxWetDry') is not None else "Mux Wet/Dry : Non disponible")
            print(f"SR Wet/Dry : {res['SRWetDry']:.3f}" if res.get('SRWetDry') is not None else "SR Wet/Dry : Non disponible")

            print("\n")


    def launch_tire_analysis(self):
        self.analysis_window = TireAnalysis(self.results, self.tire_path)
        self.analysis_window.show()

    def goBack(self):
        self.parent.show()
        self.close()
