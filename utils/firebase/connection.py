from firebase import firebase

# constants
TMP_FILE_LOCATION = "./tmp.txt"
# data 
maskData = {
    'blanco': 0,
    'gris': 0,
    'naranjaClaro': 0,
    'naranjaOscuro': 0,
    'marron': 0,
    'negro': 0
}
eyesData = {
    'naranja': 0,
    'amarillo': 0,
    'verde': 0,
    'celeste': 0
}

principalEyeColour = "-"

def generateMaskObject(file):
    while True:
        line = file.readline()
        if line == '\n':
            break
        slice = line.split(':') 
        name = slice[0]
        value = slice[1].split('\n')[0]
        maskData[name] = float(value)

def generateEyeObject(file):
    while True:
        line = file.readline()
        slice = line.split(':') 
        name = slice[0]
        if len(slice) == 1:
            return name
        else:
            value = slice[1].split('\n')[0]
            eyesData[name] = float(value)



def postDataToFirebase(idCat, image):
    #open tmp file to get the data
    tmpFileManager = open(TMP_FILE_LOCATION, 'r')
    #connect to firebase database
    connection = firebase.FirebaseApplication('https://vector-cat.firebaseio.com/', None)
    #get the data from tmp file
    generateMaskObject(tmpFileManager)
    principalEyeColour = generateEyeObject(tmpFileManager)
    #close tmp file
    tmpFileManager.close()

    #print(maskData)
    #print(eyesData)
    #print(principalEyeColour)

    #create object to post
    dataToDB =  {
        'idCat':idCat,
        'imageURL': image,
        'Face': {
            'blanco':maskData['blanco'],
            'gris':maskData['gris'],
            'naranjaClaro':maskData['naranjaClaro'],
            'naranjaOscuro':maskData['naranjaOscuro'],
            'marron':maskData['marron'],
            'negro':maskData['negro'],
        },
        'Eyes': {
            'naranja':eyesData['naranja'],
            'amarillo':eyesData['amarillo'],
            'verde':eyesData['verde'],
            'celeste':eyesData['celeste'],
            'principalEyeColour':principalEyeColour,
        },
    }
    #post object to firebase
    result = connection.post('/missing/cats/',dataToDB)
    #print answer from firebase
    print(result)