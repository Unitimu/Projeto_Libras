import cv2
from handtracking import HandDetector
import pickle

# save the model to disk
filename = "finalized_model.sav"

# load the model from disk
loaded_model = pickle.load(open(filename, "rb"))


def main():

    cam_capture = cv2.VideoCapture(0)

    while True:

        _, img = cam_capture.read()
        detector = HandDetector()
        img = detector.find_hands(img)
        teste = detector.find_position(img)

        if teste != []:
            letra_predita = loaded_model.predict(np.array(teste).reshape(1, -1))
            print(letra_predita)
            print(teste)

            cv2.putText(
                img,
                letra_predita[0],
                (150, 250),
                cv2.FONT_HERSHEY_TRIPLEX,
                4.0,
                (139, 139, 0),
                lineType=cv2.LINE_AA,
            )

        cv2.imshow("frame", img)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cam_capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

cv2.destroyAllWindows()
