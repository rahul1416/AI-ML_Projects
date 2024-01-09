# MNIST-Digit-Recognizer-Streamlit
- Install the requirements
  ```sh
  pip install -r requirements.txt
  ```
- Train the model.
  ```sh
  python3 src/model.py
  ```
- run the app
  ```sh
  streamlit run src/app.py
  ```
- model fine_tuning with updated data
  ```sh
  python3 src/fine_tune.py
  ```
In fine_tune model i have used the updated data('based on the feedback of user for incorrected datas and stored it into my training data to do the training
