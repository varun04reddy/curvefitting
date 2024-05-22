import tkinter as tk
from tkinter import Button, Radiobutton, StringVar, Checkbutton, IntVar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import curve_fitter as cf

class PlotApplication:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive Curve Fitting")

        self.figure = Figure(figsize=(6, 4))
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Click to add points")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.points = []

        self.method_var = StringVar(value="lls")
        methods = [("Linear Least Squares", "lls"), ("Total Least Squares", "tls"), ("RANSAC", "ransac")]
        for text, method in methods:
            Radiobutton(master, text=text, variable=self.method_var, value=method).pack(anchor=tk.W)

        # Toggle for Eigenvectors
        self.eigenvector_var = IntVar()
        Checkbutton(master, text="Show Eigenvectors", variable=self.eigenvector_var).pack()

        Button(master, text="Create Curve", command=self.create_curve).pack(side=tk.BOTTOM)
        Button(master, text="Clear Points", command=self.clear_points).pack(side=tk.BOTTOM)

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'ro')
            self.canvas.draw()

    def create_curve(self):
        if self.points:
            X, Y = zip(*self.points)
            method = self.method_var.get()
            
            if method == 'lls':
                m, c = cf.linear_least_squares(X, Y)
            elif method == 'tls':
                m, c = cf.total_least_squares(X, Y)
            elif method == 'ransac':
                m, c = cf.ransac(X, Y)
            
            # Plot the fitted curve
            x_vals = [min(X), max(X)]
            y_vals = [m * x + c for x in x_vals]
            self.ax.plot(x_vals, y_vals, 'b-', label=f'{method} Fit')
            
            # Optionally show eigenvectors
            if self.eigenvector_var.get():
                eigenvectors = cf.calculate_eigenvectors(X, Y)
                mean_x, mean_y = cf.mean(X), cf.mean(Y)
                for vec in eigenvectors:
                    self.ax.quiver(mean_x, mean_y, vec[0], vec[1], scale=5, color='green', alpha=0.5)

            self.ax.legend()
            self.canvas.draw()

    def clear_points(self):
        self.points.clear()
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title("Click to add points")
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = PlotApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()
