import RPi.GPIO as GPIO          
import time

#inisialisasi pin GPIO motor dc
in1 = 24
in2 = 23
in3 = 27
in4 = 22

enA = 25
enB = 17
temp1=1

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(enA,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(enB,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
pA=GPIO.PWM(enA,1000)
pB=GPIO.PWM(enB,1000)

pA.start(25)
pB.start(25)

#inisialisasi gerak robot
def jalanMaju():
    #print("Maju")
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    pA.ChangeDutyCycle(30)
    pB.ChangeDutyCycle(30)
    
def belokKiri():
    #print("Belok Kiri")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    pA.ChangeDutyCycle(50)
    pB.ChangeDutyCycle(50)
    time.sleep(.65)
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    pA.ChangeDutyCycle(30)
    pB.ChangeDutyCycle(30)
    
def belokKanan():
    #print("Belok Kanan")
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    pA.ChangeDutyCycle(50)
    pB.ChangeDutyCycle(50)
    time.sleep(.65)
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    pA.ChangeDutyCycle(30)
    pB.ChangeDutyCycle(30)

def berhenti():
    #print("Berhenti")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)


