# --------------------------------
# Librerías
# --------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from PyAstronomy import pyasl
import sys


class EW:
    def __init__(self, fits_file, csv_file, solar_csv=None, width=1.5):

        self.fits_file = fits_file
        self.csv_file = csv_file
        self.width = width
        self.solar_csv = solar_csv

        # datos
        self.wavelength = None
        self.flux = None
        self.line_data = None
        self.line_centers = None

        # estado
        self.index = 0
        self.click_points = []
        self.blending_mode = False
        self.blend_centers = []
        self.results = []

        # figura
        self.fig, self.ax = plt.subplots(figsize=(9,5))
        self.ax2 = None

        self.v_line = self.ax.axvline(0, color='m', linestyle='--', visible=False)
        self.h_line = self.ax.axhline(0, color='m', linestyle='--', visible=False)

        #Expolracion del Espectro
        self.explore_mode = False
        self.zoom_active = False
        self.zoom_xmin = None
        self.zoom_xmax = None

        self.input_mode = False
        self.input_text = ""


        self.load_data()

        if self.solar_csv is not None:
            self.load_solar_EW(self.solar_csv)

        self.show_line()

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        plt.show()





    def load_data(self):

        ext = os.path.splitext(self.fits_file)[1].lower()

        # -----------------------------
        # LECTURA FLEXIBLE DE ARCHIVOS
        # -----------------------------
        if ext in [".fits", ".fit"]:
            self.wavelength, self.flux = pyasl.read1dFitsSpec(self.fits_file)

        elif ext in [".txt", ".dat", ".ascii", ".csv"]:
            try:
                data = np.loadtxt(self.fits_file)
            except:
                try:
                    data = np.loadtxt(self.fits_file, delimiter=",")
                except:
                    data = np.loadtxt(self.fits_file, skiprows=1)

            self.wavelength = data[:, 0]
            self.flux = data[:, 1]

        else:
            raise ValueError(f"Formato no soportado: {ext}")
        
        self.line_data = pd.read_csv(self.csv_file)
        self.line_centers = self.line_data.iloc[:,0].values

        today = datetime.now().strftime("%Y-%m-%d")
        base = os.path.splitext(os.path.basename(self.fits_file))[0]

        self.output_dir = f"ajustes_{base}_{today}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.pdf = PdfPages(os.path.join(self.output_dir, "resultados.pdf"))

    #una sola gaussiana
    def gaussian_absorption(self, x, A, mu, sigma):
        return 1 - A*np.exp(-(x-mu)**2/(2*sigma**2))

    # Modelo de múltiples gaussianas (para blending)
    def multi_gaussian(self, x, *params):
        n = len(params)//3                 # número de gaussianas
        model = np.ones_like(x)            # continuo normalizado en 1
        for i in range(n):
            A = params[3*i]
            mu = params[3*i+1]
            sigma = params[3*i+2]
            model -= A*np.exp(-(x-mu)**2/(2*sigma**2))
        return model

    # Componente individual (clave para calcular EW en blending)
    def single_gaussian_component(self, x, A, mu, sigma):
        return A*np.exp(-(x-mu)**2/(2*sigma**2))


    # --------------------------------
    # Parámetros derivados
    # --------------------------------
    # FWHM desde sigma
    def compute_fwhm(self, sigma):
        return 2*np.sqrt(2*np.log(2))*sigma

    # Área bajo la gaussiana
    def compute_area(self, A,sigma):
        return A*sigma*np.sqrt(2*np.pi)

    # Equivalent Width desde modelo
    def compute_EW_model(self, x, model):
        return np.trapezoid((1 - model), x) * 1000   # en mÅ

    # Estimación de ruido
    def estimate_noise(self, y, model):
        return np.std(y - model)

    # Chi cuadrado reducido (calidad del fit)
    def compute_reduced_chi2(self, y, model, sigma, n_params):
        sigma = max(sigma, 1e-3)   # evita división por 0
        return np.sum(((y - model)/sigma)**2) / (len(y) - n_params)
    


    #Asi hacemos que las longitudes de onda coincidan, evitamos problema de incongruencias.
    def get_closest_ew(self, wavelength, tol=0.01):
        """
        Retorna el EW solar más cercano en longitud de onda.

        Parámetros:
        -----------
        wavelength : float
            Longitud de onda de la línea actual
        tol : float
            Tolerancia en Å para considerar match

        Retorna:
        --------
        float : EW solar o np.nan si no hay match
        """

        if not hasattr(self, "solar_EW_map") or len(self.solar_EW_map) == 0:
            return np.nan

        # convertir a arrays consistentes
        keys = np.array(list(self.solar_EW_map.keys()))
        values = np.array(list(self.solar_EW_map.values()))

        # buscar más cercano
        diffs = np.abs(keys - wavelength)
        idx = np.argmin(diffs)

        # check tolerancia
        if diffs[idx] <= tol:
            return values[idx]
        else:
            return np.nan



    def build_continuum(self, x, click_points):
        if len(click_points) < 2:
            return None
        x_pts = [p[0] for p in click_points]
        y_pts = [p[1] for p in click_points]
        coeffs = np.polyfit(x_pts, y_pts, 1)
        return np.polyval(coeffs, x)
    



    #ESTA ES LA FUNCION NUEVA PARA EL LOAD DEL SOL
    def load_solar_EW(self, solar_csv):

        df_sun = pd.read_csv(solar_csv)

        print("Columnas disponibles:", df_sun.columns)

        # detectar nombre automáticamente
        if "ew_Sun" in df_sun.columns:
            ew_col = "ew_Sun"
        elif "ew" in df_sun.columns:
            ew_col = "ew"
        else:
            raise ValueError("No se encontró columna de EW válida")

        self.solar_EW_map = dict(zip(df_sun["wavelength"], df_sun[ew_col]))
        self.solar_df = df_sun


    def get_ref_line(self, wavelength, tol=0.01):

        if not hasattr(self, "solar_df"):
            return None

        diffs = np.abs(self.solar_df["wavelength"].values - wavelength)
        idx = np.argmin(diffs)

        if diffs[idx] <= tol:
            return self.solar_df.iloc[idx]
        else:
            return None





    # --------------------------------
    # Mostrar línea
    # --------------------------------
    # Dibuja la línea actual en pantalla
    def show_line(self):

        self.ax.clear()

        # --------------------------------
        # CROSSHAIR
        # --------------------------------
        self.v_line = self.ax.axvline(0, color='m', linestyle='--', linewidth=0.8, visible=False)
        self.h_line = self.ax.axhline(0, color='m', linestyle='--', linewidth=0.8, visible=False)

        # eliminar zoom anterior (subplot derecho)
        if self.ax2 is not None:
            try:
                self.ax2.remove()
            except:
                pass
            self.ax2 = None

        center = self.line_centers[self.index]

        # --------------------------------
        # DEFINIR RANGO (ZOOM O NORMAL)
        # --------------------------------
        if self.zoom_active and self.zoom_xmin is not None:
            xmin, xmax = self.zoom_xmin, self.zoom_xmax
        else:
            xmin, xmax = center - self.width, center + self.width

        mask = (self.wavelength > xmin) & (self.wavelength < xmax)
        x = self.wavelength[mask]
        y = self.flux[mask]

        # --------------------------------
        # PLOT PRINCIPAL
        # --------------------------------


        # --------------------------------
        # PLOT PRINCIPAL
        # --------------------------------
        self.ax.plot(x, y, 'k-')

        # --------------------------------
        # CONTINUO DE REFERENCIA
        # --------------------------------
        if hasattr(self, "solar_df"):

            ref_row = self.get_ref_line(center)

            if ref_row is not None:

                x1 = ref_row["wave_left"]
                x2 = ref_row["wave_right"]
                y1 = ref_row["left_continuum"]
                y2 = ref_row["right_continuum"]

                # SOLO dibujar si está dentro del rango visible
                if (x1 < xmax and x2 > xmin):

                    x_ref = np.linspace(x1, x2, 100)
                    coeffs = np.polyfit([x1, x2], [y1, y2], 1)
                    y_ref = np.polyval(coeffs, x_ref)

                    self.ax.plot(x_ref, y_ref, 'r-', lw=2, label="Continuo ref")
                    self.ax.plot([x1, x2], [y1, y2], 'ro')

                    self.ax.legend()

                    self.ax.text(
                        0.75, 0.9,
                        "Ref",
                        transform=self.ax.transAxes,
                        color="red",
                        fontsize=10
                    )



        # línea central SIEMPRE visible
        self.ax.axvline(center, color="green", linestyle=":")

        # límites correctos
        self.ax.set_xlim(xmin, xmax)

        if len(y) > 0:
            self.ax.set_ylim(np.min(y)*0.98, np.max(y) + 0.1)

        # --------------------------------
        # TEXTO DE MODO
        # --------------------------------
        mode_text = "BLENDING" if self.blending_mode else "NORMAL"

        self.ax.text(
            0.02, 0.95,
            f"Modo: {mode_text}",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7)
        )

        # --------------------------------
        # INDICADOR DE ZOOM ACTIVO
        # --------------------------------
        if self.zoom_active:
            self.ax.text(
                0.02, 0.88,
                f"ZOOM: {xmin:.2f} - {xmax:.2f}",
                transform=self.ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

        # --------------------------------
        # INPUT PANEL (si está activo)
        # --------------------------------
        if self.input_mode:
            self.ax.text(
                0.02, 0.05,
                self.input_text,
                transform=self.ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.9)
            )

        # --------------------------------
        # TÍTULO Y GRID
        # --------------------------------
        self.ax.set_title(f"Línea {self.index+1} λ={center:.3f}")
        self.ax.grid()

        plt.draw()


    # --------------------------------
    # FIT
    # --------------------------------
    # Hace el ajuste automático (normal o blending)
    def auto_fit(self):

        

        target_idx = None
        mu_fit = None
        sigma_fit = None

        # necesita al menos 2 puntos para definir región
        if len(self.click_points) < 2:
            return

        # define región seleccionada
        x1, x2 = self.click_points[0][0], self.click_points[1][0]
        xmin, xmax = min(x1,x2), max(x1,x2)

        mask = (self.wavelength >= xmin) & (self.wavelength <= xmax)
        x = self.wavelength[mask]
        y = self.flux[mask]

        # calcula continuo
        continuum = self.build_continuum(x , self.click_points)
        if continuum is None:
            return

        # normaliza espectro
        y_norm = y / continuum


        # --------------------------------
        # CASO BLENDING
        # --------------------------------
        if self.blending_mode and len(self.blend_centers) > 0:

            p0 = []
            lower, upper = [], []

            for mu in self.blend_centers:
                p0 += [0.5, mu, 0.1]
                lower += [0, mu-0.2, 0.01]
                upper += [1.5, mu+0.2, 0.5]

            try:
                popt,_ = curve_fit(self.multi_gaussian, x, y_norm, p0=p0, bounds=(lower, upper))
            except:
                print("Fit falló")
                return

            model = self.multi_gaussian(x,*popt)

            EW_components = []
            mus = []
            sigmas = []

            for i in range(len(popt)//3):
                A = popt[3*i]
                mu = popt[3*i+1]
                sigma = popt[3*i+2]

                component = self.single_gaussian_component(x, A, mu, sigma)
                EW_i = np.trapezoid(component, x) * 1000

                EW_components.append(EW_i)
                mus.append(mu)
                sigmas.append(sigma)

                print(f"Comp {i+1}: μ={mu:.4f}, EW={EW_i:.2f} mÅ")

            target_center = self.line_centers[self.index]
            distances = [abs(mu - target_center) for mu in mus]
            target_idx = np.argmin(distances)

            EW_target = EW_components[target_idx]
            mu_fit = mus[target_idx]
            sigma_fit = sigmas[target_idx]

            FWHM = self.compute_fwhm(sigma_fit)

            text = ""
            for i in range(len(popt)//3):
                text += f"{i+1}: μ={mus[i]:.4f}, EW={EW_components[i]:.2f}\n"

            text += f"\nEW(target) = {EW_target:.2f}"


        # --------------------------------
        # CASO NORMAL
        # --------------------------------
        else:
            A_guess = 1 - np.min(y_norm)
            mu_guess = x[np.argmin(y_norm)]

            try:
                popt,_ = curve_fit(
                    self.gaussian_absorption, x, y_norm,
                    p0=[A_guess, mu_guess, 0.1],
                    bounds=([0, mu_guess-0.2, 0.01],[1.5, mu_guess+0.2, 0.5])
                )
            except:
                print("Fit falló")
                return

            model = self.gaussian_absorption(x,*popt)

            mu_fit, sigma_fit = popt[1], popt[2]

            FWHM = self.compute_fwhm(sigma_fit)

            EW_target = np.trapezoid(1 - y_norm, x) * 1000

            text = f"μ={mu_fit:.4f}\nEW={EW_target:.2f}"


        # calidad del ajuste
        chi2 = self.compute_reduced_chi2(
            y_norm,
            model,
            self.estimate_noise(y_norm, model),
            len(popt)
        )

        row = self.line_data.iloc[self.index]

        element = row.get("element", "")
        species = row.get("species", "")
        ep = row.get("ep", np.nan)
        gf = row.get("gf", np.nan)

        # --------------------------------
        # GUARDAR RESULTADOS
        # --------------------------------

        #GUARDAR DATOS DEL SOL
        ew_sun = np.nan

        if hasattr(self, "solar_EW_map"):
            ew_sun = self.get_closest_ew(self.line_centers[self.index])

        self.results = [
            r for r in self.results
            if r["wavelength"] != self.line_centers[self.index]
        ]

        self.results.append({
            "wavelength": self.line_centers[self.index],
            "wave_left": xmin,
            "wave_right": xmax,
            "left_continuum": self.click_points[0][1],
            "right_continuum": self.click_points[1][1],
            "element": element,
            "species": species,
            "ep": ep,
            "gf": gf,
            "ew_ref": ew_sun,
            "ew": EW_target,
            "FWHM": FWHM,
            "Chi2R": chi2,
            "hpf": 0
        })

        print(f"[AUTO-SAVE] λ={self.line_centers[self.index]:.3f} EW={EW_target:.2f}")

        # gráfico principal
        self.ax.plot(x, y_norm, 'ko', ms=3)
        self.ax.plot(x, model, 'r--')
        self.ax.plot(x, continuum, 'b--')

        # --------------------------------
        # ZOOM
        # --------------------------------
        x_margin = 0.3
        mask_ext = (self.wavelength >= xmin - x_margin) & (self.wavelength <= xmax + x_margin)

        x_ext = self.wavelength[mask_ext]
        y_ext = self.flux[mask_ext]

        continuum_ext = self.build_continuum(x_ext, self.click_points)
        if continuum_ext is None:
            return

        y_ext_norm = y_ext / continuum_ext

        self.ax2 = self.fig.add_axes([0.6, 0.15, 0.35, 0.7])
        self.ax2.plot(x_ext, y_ext_norm, 'ko', ms=3)
        self.ax2.plot(x, model, 'r--')

        self.ax2.set_xlim(xmin - x_margin, xmax + x_margin)
        self.ax2.set_title("Zoom")
        self.ax2.grid()

        self.ax2.text(
            0.95,0.05,
            text + f"\nχ²={chi2:.2f}",
            transform=self.ax2.transAxes,
            ha='right', va='bottom',
            bbox=dict(facecolor="white", alpha=0.8)
        )

        # guardar en PDF
        self.pdf.savefig(self.fig)

        plt.draw()


    # --------------------------------
    # EVENTOS (teclado)
    # --------------------------------
    def on_key(self, event):


        if self.input_mode:

            if event.key == "enter":
                try:
                    parts = self.input_text.replace("=", " ").split()

                    xmin = float(parts[1])
                    xmax = float(parts[3])

                    self.zoom_xmin = xmin
                    self.zoom_xmax = xmax
                    self.zoom_active = True

                    print(f"[ZOOM] {xmin} - {xmax}")

                    # mover al lambda más cercano
                    center = (xmin + xmax)/2
                    idx = np.argmin(np.abs(self.line_centers - center))
                    self.index = idx

                except:
                    print("Formato inválido. Usa xmin=... xmax=...")

                self.input_mode = False
                self.input_text = ""
                self.show_line()
                return

            elif event.key == "backspace":
                self.input_text = self.input_text[:-1]

            elif len(event.key) == 1:
                self.input_text += event.key

            # dibujar panel
            self.ax.text(
                0.02, 0.05,
                self.input_text,
                transform=self.ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8)
            )

            plt.draw()
            return

        # siguiente línea
        if event.key == "n":
            self.index = min(len(self.line_centers)-1, self.index+1)
            self.click_points.clear()
            self.show_line()

        # línea anterior
        if event.key == "p":
            self.index = max(0, self.index-1)
            self.click_points.clear()
            self.show_line()

        # selección de región (modo normal)
        if event.key == "k" and not self.blending_mode:
            self.click_points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, "go")

            if len(self.click_points) >= 2:
                self.auto_fit()

        # activar/desactivar blending
        if event.key == "b":
            self.blending_mode = not self.blending_mode
            self.click_points.clear()
            self.blend_centers.clear()

            print("Blending:", self.blending_mode)
            self.show_line()

        # definir región de blending
        if event.key == "d" and self.blending_mode:
            self.click_points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, "go")

            if len(self.click_points) == 2:
                x1, x2 = self.click_points[0][0], self.click_points[1][0]
                xmin, xmax = min(x1,x2), max(x1,x2)

                self.ax.axvline(xmin, color='red')
                self.ax.axvline(xmax, color='red')
                self.ax.axvspan(xmin, xmax, color='red', alpha=0.1)

        # agregar centros de líneas en blending
        if event.key == "g" and self.blending_mode:
            self.blend_centers.append(event.xdata)
            self.ax.axvline(event.xdata, color="purple", ls="--")

        # ejecutar fit en blending
        if event.key == "enter" and self.blending_mode:
            self.auto_fit()

        # reset completo
        if event.key == "r":
            self.click_points.clear()
            self.blend_centers.clear()
            self.blending_mode = False

            self.zoom_active = False
            self.zoom_xmin = None
            self.zoom_xmax = None

            print("Reset completo")
            self.show_line()





        if event.key == "x":
            self.explore_mode = not self.explore_mode
            self.click_points.clear()

            print("Modo exploración:", self.explore_mode)
            self.show_line()


        if event.key == "c" and self.explore_mode:
            self.input_mode = True
            self.input_text = ""
            print("Input: xmin=... xmax=...")


        if event.key == "a":
            self.input_mode = False
            self.input_text = ""
            print("Input cancelado")
            self.show_line()


        # cerrar
        if event.key == "q":
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(self.output_dir, "resultados.csv")
            df.to_csv(csv_path, index=False)

            self.pdf.close()

            print(f"Resultados guardados en {csv_path}")
            plt.close()

        plt.draw()


    # --------------------------------
    # Movimiento del mouse (crosshair)
    # --------------------------------
    def on_mouse_move(self, event):

        if event.inaxes:
            self.v_line.set_visible(True)
            self.h_line.set_visible(True)

            self.v_line.set_xdata([event.xdata, event.xdata])
            self.h_line.set_ydata([event.ydata, event.ydata])

        else:
            self.v_line.set_visible(False)
            self.h_line.set_visible(False)

        self.fig.canvas.draw_idle()


    @staticmethod
    def run():
        import sys

        # -------------------------
        # MODO ARGUMENTOS
        # -------------------------
        if len(sys.argv) >= 3:
            fits_file = sys.argv[1]
            csv_file = sys.argv[2]

            width = 1.5
            solar_csv = None

            if len(sys.argv) >= 4:
                try:
                    width = float(sys.argv[3])
                except:
                    print("Invalid width → using default 1.5")

            if len(sys.argv) >= 5:
                solar_csv = sys.argv[4]

            return EW(fits_file, csv_file, solar_csv, width)

        # -------------------------
        # MODO INTERACTIVO
        # -------------------------
        print("\n=== SpelacEW Interactive Mode ===\n")

        spectra = input("spectrum: ").strip()
        lines = input("line_list (.csv): ").strip()

        if spectra == "" or lines == "":
            print("ERROR: spectrum and line_list are required.")
            return None

        ref = input("(optional) ref_spectrum (.csv): ").strip() or None

        width_input = input("(optional) width (default=1.5): ").strip()
        width = float(width_input) if width_input else 1.5

        print("\nLoading...\n")

        return EW(spectra, lines, ref, width)


    


if __name__ == "__main__":

    # -------------------------
    # MODO CON ARGUMENTOS
    # -------------------------
    if len(sys.argv) >= 3:

        fits_file = sys.argv[1]
        csv_file = sys.argv[2]

        width = 1.5
        solar_csv = None

        if len(sys.argv) >= 4:
            try:
                width = float(sys.argv[3])
            except:
                print("Invalid width → using default 1.5")

        if len(sys.argv) >= 5:
            solar_csv = sys.argv[4]

        EW(fits_file, csv_file, solar_csv, width)

    # -------------------------
    # MODO INTERACTIVO
    # -------------------------
    else:
        EW.run()

    






