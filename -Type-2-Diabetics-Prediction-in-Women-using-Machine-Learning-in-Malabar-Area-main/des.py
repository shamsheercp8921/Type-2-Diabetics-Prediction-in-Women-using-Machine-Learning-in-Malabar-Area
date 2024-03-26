import pickle
import streamlit as st
with open('shams.pkl','wb') as f:
    pickle.dump(modeldt,f)
    st.title(' predication diabetics')


    def predict(input_data):
        # Reshape the input data into a 2D array
        input_data = [float(i) for i in input_data]
        input_data = np.asarray(input_data).reshape(1, -1)
        prediction = modeldt.predict(input_data)
        print(prediction)
        if prediction[0] == 0:
            return 'less chance of diabetics'
        else:
            return 'no chance'


def main():
    input_1 = st.number_input('pregnency')
    input_2 = st.number_input('Glucose', 0)
    input_3= st.number_input('Heart Rate', 0)
    input_4= st.number_input('CaloriePerDay', 0, 99)
    input_5= st.number_input('Insulin', 0)
    input_6= st.number_input('BMI')
    input_7= st.number_input('DiabetesPedigreeFunction ')
    input_8 = st.number_input("Age ", 0)
 if st.button("predict"):
     result = predict([input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8])
            st.write('Prediction: ', result)

if __name__ == '__main__':
    main()