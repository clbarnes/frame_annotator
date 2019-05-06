import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog

app = tk.Tk()
app.withdraw()

answer = simpledialog.askstring("delete event", "delete which?", parent=app)
# out = messagebox.askyesno("click yes", "click yes")
# print(out)
# out = messagebox.askyesno("click no", "click no")
# print(out)
# out = messagebox.askyesno("click X", "click X")
# print(out)
