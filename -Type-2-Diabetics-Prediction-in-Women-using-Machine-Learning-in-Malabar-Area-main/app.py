import pickle
import streamlit as st

scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))
lr_model = pickle.load(open('diabetics_lr_model.pkl', 'rb'))
svc_model = pickle.load(open('diabetics_svm_model.pkl', 'rb'))
dtc_model = pickle.load(open('diabetics_dtc_model.pkl', 'rb'))


def main():
    activities = [
        'Logistic Regression Classifier',
        "Support Vector Classifier",
        "Desicion Tree Classifier"
    ]
    option = st.sidebar.selectbox(
        'Which Model would you like to use? ',
        activities)
    st.header("Diabetics Classification for Women")
    st.image("diabetics.png")
    st.subheader(option)

    pregnencies = st.number_input('pregnency')
    glucose = st.number_input('Glucose', 0)
    blood_pressure = st.number_input('Heart Rate', 0)
    skin_thickness = st.number_input('CaloriePerDay', 0, 99)
    insulin = st.number_input('Insulin', 0)
    bmi = st.number_input('BMI')
    diabetes_pedigree_function = st.number_input(
        "DiabetesPedigreeFunction ")
    age = st.number_input("Age ", 0)

    if st.button("classify"):

        inputs = scaler_model.transform([[
            pregnencies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age
        ]])

        if option == "Logistic Regression Classifier":
            if lr_model.predict(inputs) > 0.5:
                st.error(
                    f"Thers is a chance of diabetics with model confidence of {lr_model.predict_proba(inputs)[0][1] * 100}"
                )
            else:
                st.success(
                    f"Thers is no chance of diabetics with model confidence of {lr_model.predict_proba(inputs)[0][0] * 100}"
                )
        elif option == "Support Vector Classifier":
            if lr_model.predict(inputs) > 0.5:
                st.error(
                    f"Thers is a chance of diabetics with model confidence of {svc_model.predict_proba(inputs)[0][1] * 100}"
                )
            else:
                st.success(
                    f"Thers is no chance of diabetics with model confidence of {svc_model.predict_proba(inputs)[0][0] * 100}"
                )
        else:
            if lr_model.predict(inputs) > 0.5:
                st.error(
                    f"Thers is a chance of diabetics with model confidence of {dtc_model.predict_proba(inputs)[0][1] * 100}"
                )
            else:
                st.success(
                    f"Thers is no chance of diabetics with model confidence of {dtc_model.predict_proba(inputs)[0][0] * 100}"
                )


main()
