import pandas as pd
import re
import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from scipy.stats import linregress

class TireAnalysis(QWidget):
    def __init__(self, results, tire_path):
        super().__init__()
        self.results = results
        self.tire_path = tire_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Longitudinal Grip Dependency')
        self.layout = QVBoxLayout(self)
        
        # Create a pyqtgraph PlotWidget and add it to the layout
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.plot_widget)
        
        self.setLayout(self.layout)
        self.plot_longitudinal_grip_dependency()


    def results_to_csv(self):
        df = pd.DataFrame(self.results).T
        df.reset_index(inplace=True)
        df.rename(columns={'level_0': 'Condition', 'level_1': 'Subcondition'}, inplace=True)
        df = df.drop(columns=['SR_opt', 'Mux_opt'])
        df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').abs()

        if 'mux_max' in df.columns:
            df['mux_max'] = df['mux_max'].abs()
        if 'sr_opt' in df.columns:
            df['sr_opt'] = df['sr_opt'].abs()
        
        tire_id_match = re.search(r'/([^/]+)/[^/]+$', self.tire_path)
        full_tire_id = tire_id_match.group(1) if tire_id_match else 'Unknown'
        
        id_match = re.match(r'(\d{2}-\d{3}-[A-Z]{2})', full_tire_id)
        tire_id = id_match.group(1) if id_match else 'Unknown'
        
        df.insert(0, 'Tire ID', tire_id)
        file_path = 'results.csv'
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, df]).drop_duplicates().reset_index(drop=True)
        else:
            combined_df = df
        
        combined_df.to_csv(file_path, index=False)
        return df 



    def plot_longitudinal_grip_dependency(self):
        df = self.results_to_csv()
        conditions = ['FZ40', 'FZ80', 'FZ120']
        subconditions = ['ND', 'NW']
        indicators = ['mux_max', 'sr_opt']

        self.plot_widget.clear()

        for i, indicator in enumerate(indicators):
            for j, subcondition in enumerate(subconditions):
                plot = self.plot_widget.addPlot(row=i, col=j)
                x_vals = []
                y_vals = []
                for k, condition in enumerate(conditions):
                    data = df.loc[(df['Condition'] == condition) & (df['Subcondition'] == subcondition), indicator].values
                    # Convertir les données en nombres avant de filtrer
                    data = pd.to_numeric(data, errors='coerce')
                    # Filtrer les données pour ne garder que les valeurs finies
                    data = data[np.isfinite(data)]
                    if len(data) > 0:
                        plot.plot([k] * len(data), data, pen=None, symbol='o', label=f'{condition} - {subcondition}')
                        x_vals.append(k)
                        y_vals.append(np.mean(data))
                
                # Effectuer la régression linéaire
                if len(x_vals) > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
                    x_range = np.linspace(min(x_vals), max(x_vals), 100)
                    y_range = slope * x_range + intercept
                    plot.plot(x_range, y_range, pen='r', name='Régression linéaire')
                
                plot.setTitle(f'{indicator} pour {subcondition}')
                plot.setLabel('left', indicator)
                plot.setLabel('bottom', 'Condition')
                plot.getAxis('bottom').setTicks([list(enumerate(conditions))])
                plot.addLegend()
                plot.showGrid(x=True, y=True)

                # print les coeficient a de la pente 
                print(f'Pente de la régression linéaire pour {indicator} - {subcondition}: {slope}')

                # affiche moi le 


        self.plot_widget.nextRow()