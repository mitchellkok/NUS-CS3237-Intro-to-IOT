from typing import runtime_checkable
import paho.mqtt.client as mqtt
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session

classes = ["daisy", "danadelion", "roses", "sunflowers", "tulips"]

MODEL_NAME = "flowers.hd5"
session = tf.compat.v1.Session(graph = tf.compat.v1.Graph())

with session.graph.as_default():
    set_session(session)
    model = load_model(MODEL_NAME)
    session.run(tf.compat.v1.global_variables_initializer())

def on_connect(client, userdata, flags, rc):
    if rc==0:
        print("Successfully connected to broker.")
        client.subscribe("Group_B3/IMAGE/classify")
    else:
        print("Connection failed with code: %d." % rc)

def classify_flower(filename, data):
    print("Start classifying")

    with session.graph.as_default():    # use session.graph.as_default to preserve graphs
        set_session(session)            # set the tensorflow session
        result = model.predict(data)    # predict class using model, based on data
        themax = np.argmax(result)      # obtain the index of the largest value in result

    # convert data into JSON serializable datatypes
    win = int(themax)
    score = float(result[0][themax])
    print("Done.")

    # return prediction results
    return {"filename": filename, "prediction": classes[win], "score" : score, "index": win}

def on_message(client, userdata, msg):
    # payload is in msg. we convert it back to a Python dictionary
    recv_dict = json.loads(msg.payload)

    # recreate the data
    img_data = np.array(recv_dict["data"])
    result = classify_flower(recv_dict["filename"], img_data)   # classify data

    print("Sending results: ", result)
    client.publish("Group_B3/IMAGE/predict", json.dumps(result))

def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(hostname)
    client.loop_start()
    return client

def main():
    setup("192.168.1.1")    # edit according to IP address
    while True:
        pass

if __name__ == '__main__':
    main()