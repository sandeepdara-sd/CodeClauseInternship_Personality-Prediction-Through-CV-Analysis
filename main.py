import os
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import filedialog, ttk
import tkinter.font as font
from functools import partial
from pyresparser import ResumeParser
from sklearn import datasets, linear_model 
import spacy
import subprocess

# Download spaCy English model
try:
    spacy.load('en_core_web_sm')
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

class train_model:
    
    def train(self):
        data = pd.read_csv('training_dataset.csv')
        array = data.values

        for i in range(len(array)):
            if array[i][0] == "Male":
                array[i][0] = 1
            else:
                array[i][0] = 0

        df = pd.DataFrame(array)

        maindf = df[[0, 1, 2, 3, 4, 5, 6]]
        mainarray = maindf.values

        temp = df[7]
        train_y = temp.values
        
        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.mul_lr.fit(mainarray, train_y)
        
    def test(self, test_data):
        try:
            test_predict = list()
            for i in test_data:
                test_predict.append(int(i))
            y_pred = self.mul_lr.predict([test_predict])
            return y_pred
        except:
            print("All Factors For Finding Personality Not Entered!")


def check_type(data):
    if type(data) == str or type(data) == str:
        return str(data).title()
    if type(data) == list or type(data) == tuple:
        str_list = ""
        for i, item in enumerate(data):
            str_list += item + ", "
        return str_list

def prediction_result(top, aplcnt_name, cv_path, personality_values):
    top.withdraw()
    applicant_data = {"Candidate Name": aplcnt_name.get(), "CV Location": cv_path}
    
    age = personality_values[1]
    
    print("\n############# Candidate Entered Data #############\n")
    print(applicant_data, personality_values)
    
    personality = model.test(personality_values)
    print("\n############# Predicted Personality #############\n")
    print(personality)
    
    # Corrected spaCy initialization
    try:
        data = ResumeParser(cv_path, spacy_model='en_core_web_sm').get_extracted_data()
    except Exception as e:
        print(f"Error while parsing resume: {e}")
        data = {}

    try:
        del data['name']
        if len(data['mobile_number']) < 10:
            del data['mobile_number']
    except:
        pass
    
    print("\n############# Resume Parsed Data #############\n")

    for key in data.keys():
        if data[key] is not None:
            print('{} : {}'.format(key, data[key]))
    
    result = Tk()
    result.title("Predicted Personality")
    result.geometry("800x600")
    result.configure(background='#f0f0f0')
    
    titleFont = font.Font(family='Arial', size=32, weight='bold')
    Label(result, text="Result - Personality Prediction", foreground='#006633', bg='#f0f0f0', font=titleFont, pady=20).pack(fill=BOTH)
    
    Label(result, text=str('{} : {}'.format("Name:", aplcnt_name.get())).title(), foreground='#333333', bg='#f0f0f0', anchor='w', padx=20).pack(fill=BOTH)
    Label(result, text=str('{} : {}'.format("Age:", age)), foreground='#333333', bg='#f0f0f0', anchor='w', padx=20).pack(fill=BOTH)
    for key in data.keys():
        if data[key] is not None:
            Label(result, text=str('{} : {}'.format(check_type(key.title()), check_type(data[key]))), foreground='#333333', bg='#f0f0f0', anchor='w', width=60, padx=20).pack(fill=BOTH)
    Label(result, text=str("Predicted Personality: " + personality).title(), foreground='#333333', bg='#f0f0f0', anchor='w', padx=20).pack(fill=BOTH)
    
    quitBtn = ttk.Button(result, text="Exit", command=result.destroy)
    quitBtn.pack(pady=10)

    terms_mean = """
# Openness:
    People who like to learn new things and enjoy new experiences usually score high in openness. Openness includes traits like being insightful and imaginative and having a wide variety of interests.

# Conscientiousness:
    People that have a high degree of conscientiousness are reliable and prompt. Traits include being organised, methodic, and thorough.

# Extraversion:
    Extraversion traits include being; energetic, talkative, and assertive (sometime seen as outspoken by Introverts). Extraverts get their energy and drive from others, while introverts are self-driven get their drive from within themselves.

# Agreeableness:
    As it perhaps sounds, these individuals are warm, friendly, compassionate and cooperative and traits include being kind, affectionate, and sympathetic. In contrast, people with lower levels of agreeableness may be more distant.

# Neuroticism:
    Neuroticism or Emotional Stability relates to degree of negative emotions. People that score high on neuroticism often experience emotional instability and negative emotions. Characteristics typically include being moody and tense.    
"""
    
    Label(result, text=terms_mean, foreground='#006633', bg='#f0f0f0', anchor='w', justify=LEFT, padx=20).pack(fill=BOTH)

    result.mainloop()


def predict_person():
    root.withdraw()
    top = Toplevel()
    top.geometry('800x600')
    top.title("Apply For A Job")
    top.configure(background='#333333')
    
    titleFont = font.Font(family='Helvetica', size=30, weight='bold')
    Label(top, text="Personality Prediction", foreground='#FF4500', bg='#333333', font=titleFont, pady=20).pack()

    job_list = ('Select Job', '101-Developer at TTC', '102-Chef at Taj', '103-Professor at MIT')
    job = StringVar(top)
    job.set(job_list[0])

    labels = [
        "Applicant Name", "Age", "Gender", "Upload Resume", 
        "Enjoy New Experience or thing(Openness)", 
        "How Often You Feel Negativity(Neuroticism)", 
        "Wishing to do one's work well and thoroughly(Conscientiousness)", 
        "How much would you like to work with your peers(Agreeableness)", 
        "How outgoing and social interaction you like(Extraversion)"
    ]

    entries = []

    for i, label in enumerate(labels):
        Label(top, text=label, foreground='#ffffff', bg='#333333').place(x=70, y=130 + 30 * i)

    sName = Entry(top)
    sName.place(x=450, y=130, width=160)
    age = Entry(top)
    age.place(x=450, y=160, width=160)
    gender = IntVar()
    R1 = Radiobutton(top, text="Male", variable=gender, value=1, padx=7)
    R1.place(x=450, y=190)
    R2 = Radiobutton(top, text="Female", variable=gender, value=0, padx=3)
    R2.place(x=540, y=190)
    
    cvBtn = ttk.Button(top, text="Select File", command=lambda: OpenFile(cvBtn))
    cvBtn.place(x=450, y=220, width=160)

    openness = Entry(top)
    openness.insert(0, '1-10')
    openness.place(x=450, y=250, width=160)
    neuroticism = Entry(top)
    neuroticism.insert(0, '1-10')
    neuroticism.place(x=450, y=280, width=160)
    conscientiousness = Entry(top)
    conscientiousness.insert(0, '1-10')
    conscientiousness.place(x=450, y=310, width=160)
    agreeableness = Entry(top)
    agreeableness.insert(0, '1-10')
    agreeableness.place(x=450, y=340, width=160)
    extraversion = Entry(top)
    extraversion.insert(0, '1-10')
    extraversion.place(x=450, y=370, width=160)

    entries.extend([sName, age, openness, neuroticism, conscientiousness, agreeableness, extraversion])

    submitBtn = ttk.Button(top, text="Submit", command=lambda: prediction_result(top, sName, loc, (gender.get(), age.get(), openness.get(), neuroticism.get(), conscientiousness.get(), agreeableness.get(), extraversion.get())))
    submitBtn.place(x=350, y=400, width=200)

    top.mainloop()


def OpenFile(b4):
    global loc
    name = filedialog.askopenfilename(
        initialdir="E:/INTERN/CV_ANALYSIS/training_dataset.csv",
        filetypes=(("Document", "*.docx*"), ("PDF", "*.pdf*"), ('All files', '*')),
        title="Choose a file."
    )
    try:
        filename = os.path.basename(name)
        loc = name
    except:
        filename = name
        loc = name
    b4.config(text=filename)
    return


if __name__ == "__main__":
    model = train_model()
    model.train()

    root = Tk()
    root.geometry('800x600')
    root.configure(background='#333333')
    root.title("Personality Prediction System")
    titleFont = font.Font(family='Helvetica', size=40, weight='bold')
    homeBtnFont = font.Font(size=12, weight='bold')
    Label(root, text="Personality Prediction System", bg='#333333', font=titleFont, pady=30, fg='#FF4500').pack()
    ttk.Button(root, text="Predict Personality", style='my.TButton', command=predict_person).place(relx=0.5, rely=0.5, anchor=CENTER)

    style = ttk.Style()
    style.configure('my.TButton', foreground='white', background='#006633', font=('Helvetica', 12, 'bold'))
    
    root.mainloop()
