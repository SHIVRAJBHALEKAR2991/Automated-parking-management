import csv
import datetime
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import os

def caltime(timestamp1, timestamp2):
    time1 = datetime.datetime.strptime(timestamp1, "%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(timestamp2, "%Y-%m-%d %H:%M:%S")

    # Calculate the time difference
    time_difference = time2 - time1

    # Convert time difference to minutes
    difference_in_minutes = time_difference.total_seconds() / 60

    return difference_in_minutes

def log_vehicle_details(vehicle_number):
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    # Read existing data from the CSV file
    try:
        with open("vehicle_log.csv", 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass  # File not found, assume it's empty
    found = False
    for row in rows:
        if len(row) >= 3 and vehicle_number == row[0]:
            # Vehicle found in the log
            if row[2] == "":
                # Vehicle found with no exit time, updating exit time
                row[2] = curr_time
                duration = caltime(row[1], curr_time)
                rate = 0.50
                amt = rate * duration
                print(f"Pay parking charges Rs {int(amt) + 1} for {duration} mins at the rate of {rate} paise per min")
                found = True
                break  # Break out of the loop after updating exit time
    # If vehicle not found or not logged previously, add a new entry
    if not found:
        rows.append([vehicle_number, curr_time, ""])

    # Write updated or new data to the file
    with open("vehicle_log.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)



def detectnum(image):
    img=image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    text = result[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(text)
    return text

folderpath=("testing/exit")

files = os.listdir(folderpath)
for file in files:
    finalfile = os.path.join(folderpath,file)
    img=cv2.imread(finalfile)
    vehiclenum=detectnum(img)
    log_vehicle_details(vehiclenum)
    cv2.imshow('image',img)
    cv2.waitKey(0)

