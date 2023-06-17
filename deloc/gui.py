import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import deloc as dl
import matplotlib
from matplotlib import rcParams

rcParams['text.usetex'] = True

matplotlib.use('Qt5Agg')


class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.m_field = None
        self.m_label = None
        self.fig = None
        self.ax = None
        self.setWindowTitle("Polariton Calculation")
        self.initUI()
        self.setGeometry(100, 100, 800, 600)

    def initUI(self):
        # Create input labels and fields
        self.m_label = QLabel(r"Cavity order m (positive integer):")  # LaTeX expression in the label
        self.m_field = QLineEdit()

        self.Ec_label = QLabel(r'Cavity cutoff Ec (meV):')
        self.Ec_field = QLineEdit()

        self.E0_label = QLabel(r"Mean molecular Energy E0 (meV):")
        self.E0_field = QLineEdit()

        self.Lc_label = QLabel("Cavity Length L_c (cm):")
        self.Lc_field = QLineEdit()

        self.Delta_label = QLabel("Rabi/2=Delta (meV):")
        self.Delta_field = QLineEdit()

        self.Sigma_label = QLabel("Energetic Disorder Sigma (meV):")
        self.Sigma_field = QLineEdit()

        self.Gamma_Ex_label = QLabel("Molecular disorder Gamma_Ex (meV):")
        self.Gamma_Ex_field = QLineEdit()

        self.Gamma_L_label = QLabel("Cavity Leakage Gamma_L (meV):")
        self.Gamma_L_field = QLineEdit()

        self.q_initial_label = QLabel("q_initial:")
        self.q_initial_field = QLineEdit()

        self.q_final_label = QLabel("q_final:")
        self.q_final_field = QLineEdit()

        self.q_step_label = QLabel("q_step:")
        self.q_step_field = QLineEdit()

        # Create output labels
        self.q_inel_lp_min_label = QLabel("q_inel_lp_min:")
        self.q_inel_lp_max_label = QLabel("q_inel_lp_max:")
        self.q_inel_up_min_label = QLabel("q_inel_up_min:")
        self.q_inel_up_max_label = QLabel("q_inel_up_max:")

        # Create calculate button
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate)

        # Create the figure and axes for the plot
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1, figsize=(11, 9))
        canvas = FigureCanvas(self.fig)

        # Create layout
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.m_label)
        input_layout.addWidget(self.m_field)
        input_layout.addWidget(self.Ec_label)
        input_layout.addWidget(self.Ec_field)
        input_layout.addWidget(self.E0_label)
        input_layout.addWidget(self.E0_field)
        input_layout.addWidget(self.Lc_label)
        input_layout.addWidget(self.Lc_field)
        input_layout.addWidget(self.Delta_label)
        input_layout.addWidget(self.Delta_field)
        input_layout.addWidget(self.Sigma_label)
        input_layout.addWidget(self.Sigma_field)
        input_layout.addWidget(self.Gamma_Ex_label)
        input_layout.addWidget(self.Gamma_Ex_field)
        input_layout.addWidget(self.Gamma_L_label)
        input_layout.addWidget(self.Gamma_L_field)
        input_layout.addWidget(self.q_initial_label)
        input_layout.addWidget(self.q_initial_field)
        input_layout.addWidget(self.q_final_label)
        input_layout.addWidget(self.q_final_field)
        input_layout.addWidget(self.q_step_label)
        input_layout.addWidget(self.q_step_field)
        input_layout.addWidget(self.calculate_button)
        input_layout.addWidget(self.q_inel_lp_min_label)
        input_layout.addWidget(self.q_inel_lp_max_label)
        input_layout.addWidget(self.q_inel_up_min_label)
        input_layout.addWidget(self.q_inel_up_max_label)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(canvas)

        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(plot_layout)

        self.setLayout(main_layout)

    def calculate(self):
        # Get input values
        m = float(self.m_field.text())
        Ec = float(self.Ec_field.text())
        E0 = float(self.E0_field.text())
        Lc = float(self.Lc_field.text())
        Delta = float(self.Delta_field.text())
        Sigma = float(self.Sigma_field.text())
        Gamma_Ex = float(self.Gamma_Ex_field.text())
        Gamma_L = float(self.Gamma_L_field.text())
        q_initial = float(self.q_initial_field.text())
        q_final = float(self.q_final_field.text())
        q_step = float(self.q_step_field.text())

        # Perform calculations
        q_inel_lp_min, q_inel_lp_max, q_inel_up_min, q_inel_up_max = dl.inelastic(
            m, Ec, E0, Lc, Delta, Gamma_Ex, Gamma_L, q_initial, q_final, q_step)

        # Display results
        self.q_inel_lp_min_label.setText(f"q_inel_lp_min: {q_inel_lp_min}")
        self.q_inel_lp_max_label.setText(f"q_inel_lp_max: {q_inel_lp_max}")
        self.q_inel_up_min_label.setText(f"q_inel_up_min: {q_inel_up_min}")
        self.q_inel_up_max_label.setText(f"q_inel_up_max: No upper boundary")

        # Clear the previous plot
        self.ax.clear()
        self.ax2.clear()

        # Generate the plot
        fs = 50
        fs_text = fs - 10
        q_max_plot = q_inel_lp_max + 1e4
        q0l = np.linspace(0, q_inel_lp_min, 2900)
        q0u = np.linspace(0, q_inel_up_min, 5800)
        q0ll = np.linspace(q_inel_lp_max, q_max_plot, 2900)
        q1 = np.linspace(q_inel_up_min, q_max_plot, 17000)
        q2 = np.linspace(q_inel_lp_min, q_inel_lp_max, 17000)
        q3 = np.linspace(E0, E0, 19000)
        q4 = np.linspace(0, q_inel_lp_max, 17000)
        q5 = np.linspace(0, q_max_plot, int(1.9E+4))

        self.ax.plot(q2, dl.energy_lp_zero(m, q2, Ec, E0, Lc, Delta), linewidth=4, markersize=12, c='r')
        ### Polariton and molecular energies ###
        self.ax.plot(q1, dl.energy_up_zero(m, q1, Ec, E0, Lc, Delta), linewidth=4, markersize=12, c='b')
        self.ax.plot(q5, dl.energy_cavity(m, q5, Ec, Lc), linestyle='dashed', linewidth=4, markersize=12, c='orange')
        self.ax.plot(q5, q3, linestyle='dashed', linewidth=4, markersize=12, c='green')

        self.ax.plot(q2, dl.energy_lp_zero(m, q2, Ec, E0, Lc, Delta), linewidth=4, markersize=12, c='r')
        self.ax.plot(q0u, dl.energy_up_zero(m, q0u, Ec, E0, Lc, Delta), linewidth=8, markersize=16, c='b',
                     linestyle='dotted')
        self.ax.plot(q0l, dl.energy_lp_zero(m, q0l, Ec, E0, Lc, Delta), linewidth=8, markersize=16, c='r',
                     linestyle='dotted')
        self.ax.plot(q0ll, dl.energy_lp_zero(m, q0ll, Ec, E0, Lc, Delta), linewidth=8, markersize=16, c='r',
                     linestyle='dotted')

        ### Boundary points ###
        self.ax.plot(q_inel_up_min, dl.energy_up_zero(m, q_inel_up_min, Ec, E0, Lc, Delta), marker='o', linewidth=4,
                     markersize=fs / 3, c='k')
        self.ax.plot(q_inel_lp_min, dl.energy_lp_zero(m, q_inel_lp_min, Ec, E0, Lc, Delta), marker='o', linewidth=4,
                     markersize=fs / 3, c='k')
        self.ax.plot(q_inel_lp_max, dl.energy_lp_zero(m, q_inel_lp_max, Ec, E0, Lc, Delta), marker='o', linewidth=4,
                     markersize=fs / 3, c='k')

        # Texts
        self.ax.text(q_inel_up_min - q_final/40, dl.energy_up_zero(m, q_inel_up_min, Ec, E0, Lc, Delta) + E0 / 10,
                     r"$q_{min}^{U}$",
                     fontsize=fs_text - 10)
        self.ax.text(q_inel_lp_min + q_final/2e2, dl.energy_lp_zero(m, q_inel_lp_min, Ec, E0, Lc, Delta) - E0 / 10,
                     r'$q_{min}^{L}$',
                     fontsize=fs_text - 10)
        self.ax.text(q_inel_lp_max, dl.energy_lp_zero(m, q_inel_lp_max, Ec, E0, Lc, Delta) - E0 / 9, r'$q_{max}^{L}$',
                     fontsize=fs_text - 10)

        self.ax.set_ylabel('$E$ (meV)', fontsize=fs - 16)
        self.ax.tick_params(axis='x', labelsize=fs - 24)
        self.ax.tick_params(axis='y', labelsize=fs - 24)
        self.ax.set_ylim(E0 / 2, 3 * E0 / 2)
        self.ax.set_xlim(0, q_max_plot)
        self.ax.set_xticks(np.arange(0, q_max_plot, q_final/4))

        self.ax2.plot(q5, dl.c_l_up(m, q5, Ec, E0, Lc, Delta), linewidth=4, markersize=12, c='b')
        self.ax2.plot(q5, dl.c_l_lp(m, q5, Ec, E0, Lc, Delta), linewidth=4, markersize=12, c='r')

        self.ax2.set_xlabel(r'$q ($ cm$^{-1})$', fontsize=fs - 15)
        self.ax2.set_ylabel('Photonic weight', fontsize=fs - 15)
        # plt.legend(["LP", "UP"], loc ="center right",fontsize = fs -12 )

        self.ax2.tick_params(axis='x', labelsize=fs - 25)
        self.ax2.tick_params(axis='y', labelsize=fs - 25)

        self.ax2.set_xlim(0, q_max_plot)
        self.ax2.set_ylim(0, 1.0)

        self.ax2.set_yticks(np.arange(0.2, 1.05, 0.2))
        self.ax2.set_xticks(np.arange(0, q_max_plot, 4E+3))

        self.ax.legend(["LP", "UP", r'$E_{C}(q)$', r'$E_{M}$'], bbox_to_anchor=(0, 1, 1, 0),
                       loc="lower center", fontsize=fs - 20, ncol=4)

        self.fig.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
