import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

cara_cascada = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cuerpo_cascada = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

deteccion = False
deteccion_tiempo_detenido = None
temporizador_comienza = False
SEGUNDOS_PARA_GRABAR_DESPUES_DE_DETECCION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2gris)
    caras = cara_cascade.detectMultiScale(gris, 1.3, 5)
    cuerpos = cara_cascade.detectMultiScale(gris, 1.3, 5)

    if len(caras) + len(cuerpos) > 0:
        if deteccion:
            temporizador_comienza = False
        else:
            deteccion = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(
                f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Empezar a Grabar!")
    elif deteccion:
        if temporizador_comienza:
            if time.time() - deteccion_tiempo_detenido >= SEGUNDOS_PARA_GRABAR_DESPUES_DE_DETECCION:
                deteccion = False
                temporizador_comienza = False
                out.release()
                print('Parar de Grabar!')
        else:
            temporizador_comienza = True
            deteccion_tiempo_detenido = time.time()

    if deteccion:
        out.write(frame)

    # for (x, y, anchura, altura) en caras:
    #    cv2.rectangle(frame, (x, y), (x + anchura, y + altura), (255, 0, 0), 3)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
