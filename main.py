import cv2
import numpy as np
import math

def main():
  trainingImagesPath = [
    ".\dataset\Training\subject01.happy.jpg",
    ".\dataset\Training\subject02.normal.jpg",
    ".\dataset\Training\subject03.normal.jpg",
    ".\dataset\Training\subject07.centerlight.jpg",
    ".\dataset\Training\subject10.normal.jpg",
    ".\dataset\Training\subject11.normal.jpg",
    ".\dataset\Training\subject14.normal.jpg",
    ".\dataset\Training\subject15.normal.jpg",
  ]
  testingImagesPath = [
    ".\dataset\Testing\subject01.normal.jpg",
    ".\dataset\Testing\subject07.happy.jpg",
    ".\dataset\Testing\subject07.normal.jpg",
    ".\dataset\Testing\subject11.happy.jpg",
    ".\dataset\Testing\subject14.happy.jpg",
    ".\dataset\Testing\subject14.sad.jpg",
  ]
  imageRow = 231
  imageColumn = 195

  # For each training image, stack the row together to form a column vector R
  columnVectors = get_column_vector_set(trainingImagesPath)
  # The mean face is computed by taking the average of the training face images
  meanFace = np.mean(columnVectors, axis=0)
  show_save_mean_face(meanFace, imageRow, imageColumn)
  # We subtract the mean face from each training face and put them into a single matrix A
  A = get_normalized_column_vector_set(columnVectors, meanFace)
  # We find eigenvalues of L = A transpose x A
  L = np.matmul((A.T), A)
  # Put eigenvectors of L into a single matrix V
  _, V = np.linalg.eig(L)
  # The M largest eigenvectors of C can be found by U = AV
  U = np.matmul(A,V)
  # Display and save eigenface
  show_save_eigenface(U.T, imageRow, imageColumn)
  # For each training faces, calculate its eigenface coefficients
  trainingCoefficients = np.zeros(shape=(len(trainingImagesPath),len(trainingImagesPath)))
  for i in range(len(trainingImagesPath)):
    trainingCoefficients[i]=np.matmul(U.T,A[:,i])
  print("The Eigenface coefficients of the training images:")
  print(trainingCoefficients)
  print()

  # ----------------------------------- Face recognition -----------------------------------

  # Calculate the eigenface coefficient of each testing image
  testingColumnVectors = get_column_vector_set(testingImagesPath)
  testingNormalizedColumnVectors = get_normalized_column_vector_set(testingColumnVectors, meanFace)
  testingCoefficients = np.zeros(shape=(len(testingImagesPath),len(trainingImagesPath)))
  for i in range(len(testingImagesPath)):
    omega = np.matmul(U.T, testingNormalizedColumnVectors[:,i])
    testingCoefficients[i]=omega
  print("The Eigenface coefficients of the testing images:")
  print(testingCoefficients)
  print()

  # Calculate and print recognition result for each test image
  for i in range(len(testingCoefficients)):
    matchIndex = get_match_face(testingCoefficients[i], trainingCoefficients)
    print(testingImagesPath[i][18:] + " is match to " + trainingImagesPath[matchIndex][19:])


def get_column_vector_set(trainingImagesPath):
  #flat out images to columnvectors
  column_vector_set = 0
  for i in range(len(trainingImagesPath)):
    # read image in grayscale mode
    image = cv2.imread(trainingImagesPath[i], 0)
    image = image.reshape(1,45045)
    if i == 0:
      column_vector_set = image
    else:
      column_vector_set = np.append(column_vector_set,image,axis=0)
  return column_vector_set


def get_normalized_column_vector_set(columnVectors, meanFace):
  # normalize column vectors by subtracting the average face from them
  normalize_cv = np.ndarray(shape=(len(columnVectors),45045))
  for i in range(len(columnVectors)):
    normalize_cv[i] = columnVectors[i] - meanFace
  return normalize_cv.T


def get_euclidean_distance(a, b):
  # calculate euclidean distance between two vectors
  dist = 0
  for i in range(len(a)):
    dist += (a[i] - b[i])**2
  return math.sqrt(dist)


def get_match_face(input, dataset):
  # find the face in dataset with minimal euclidean distance from the input
  matchIndex = 0
  minDistance = float('inf')
  for i in range(len(dataset)):
    curDistance = get_euclidean_distance(input,dataset[i].reshape(8,1))
    if(curDistance < minDistance):
      minDistance = curDistance
      matchIndex = i
  return matchIndex


def show_image(image, name = "image"):
    # open the image in a new window
    # press 0 key to close images
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_save_mean_face(meanFace, imageRow, imageColumn):
  # the dimension of meanFace is N^2 x 1
  print("Displaying mean face")
  print("Press 0 to close the image")
  print()
  image = np.uint8(np.reshape(meanFace, (imageRow, imageColumn)))
  cv2.imwrite("mean face.bmp",image)
  show_image(image)


def show_save_eigenface(UT, imageRow, imageColumn):
  i = 1
  for eigenface in UT:
    print("Displaying eigenface ", i)
    print("Press 0 to close the images")
    print()
    image = np.uint8(np.reshape(eigenface, (imageRow, imageColumn)))
    cv2.imwrite("egienface"+str(i)+".bmp",image)
    i += 1
    show_image(image)


main()