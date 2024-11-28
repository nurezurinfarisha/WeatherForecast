import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import aiml as ai

import pandas as pd

# initalise the tkinter GUI
root = tk.Tk()
root.title("Rain Prediction in Australia")


root.geometry("1000x500") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
#root.resizable(0, 0) # makes the root window fixed in size.

#Styles
style = ttk.Style()

# Frame for TreeView
frame1 = tk.LabelFrame(root, text="Output")
frame1.place(height=250, width=1000, rely=0.40, relx=0)

# Frame for open file dialog
file_frame = tk.LabelFrame(root, text="Input")
file_frame.place(height=100, width=400, rely=0.10, relx=0)

# Buttons
button1 = tk.Button(file_frame, text="Browse A File", command=lambda: File_dialog())
button1.place(rely=0.65, relx=0.30)

button2 = tk.Button(file_frame, text="Load File", command=lambda: Load_excel_data())
button2.place(rely=0.65, relx=0.30)

# The file/file path text
label_file = ttk.Label(file_frame, text="No File Selected")
label_file.place(rely=0, relx=0)


## Treeview Widget
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget

#Colour tags
tv1.tag_configure('rain', background="red", foreground="white")
tv1.tag_configure('noRain', background="gray")


def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select A File",
                                          filetype=(("csv files", "*.csv"),))
    label_file["text"] = filename
    return None

def Check_String_True(x):
    condition = "Yes"
    if condition in x:
        return True
    else:
        return False


def Load_excel_data():
    """If the file selected is valid this will load the file into the Treeview"""
    file_path = label_file["text"] #This has the directory name
    try:
        df = ai.predict(file_path)
        #excel_filename = r"{}".format(file_path) 
        #if excel_filename[-4:] == ".csv":
        #    df = ai.predict(excel_filename)
        #else:
        #    df = ai.predict(excel_filename)

    #except ValueError:
        #tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        #return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None

    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name

    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        check = Check_String_True(row)
        if (check):
            tv1.insert("", "end", values=row, tags = "rain") # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
            #print(tv1.identify_element(column, 2))
        else:
            tv1.insert("", "end", values=row, tags = "noRain")
            #print(tv1.identify_element(column, 2))


    #if()
    
    return None


def clear_data():
    tv1.delete(*tv1.get_children())
    return None


root.mainloop()