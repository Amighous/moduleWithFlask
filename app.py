from flask import Flask, Response, render_template, jsonify
import cv2
import cvzone
import numpy as np

app = Flask(__name__)

# Define the positions and IDs of the parking spaces
num_spaces = 5
posList = []
space_status = {}  # Dictionary to track the status of each parking space
start_x = 50
start_y = 100
space_width = 107
space_height = 55
space_margin = 20

for i in range(num_spaces):
    y = start_y + i * (space_height + space_margin)
    for j in range(4):
        x = start_x + j * (space_width + space_margin)
        # Assign a unique ID to each parking space based on its row and column
        space_id = f"{i}-{j}"  # Format: row-column
        posList.append((space_id, x, y))
        # Initialize each space as free
        space_status[space_id] = 'free'

width, height = space_width, space_height

# Laptop camera feed
cap = cv2.VideoCapture(0)  # 0 for the default camera, change it if necessary

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

            free_spaces_count = checkParkingSpace(frame, imgDilate)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Update the count of free spaces
            if free_spaces_count is not None:
                app.config['free_spaces_count'] = free_spaces_count

def checkParkingSpace(img, imgPro):
    spaceCounter = 0
    free_spaces_ids = []  # List to store IDs of free spaces

    for space_id, x, y in posList:
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)  # Green color for free space
            thickness = 5
            spaceCounter += 1
            free_spaces_ids.append(space_id)  # Append ID of free space
            space_status[space_id] = 'free'  # Update status
        else:
            color = (0, 0, 255)  # Red color for occupied space
            thickness = 2
            space_status[space_id] = 'occupied'  # Update status

        # Draw rectangle around parking space
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
        # Add count text
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    # Add text showing number of free spaces
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))

    return free_spaces_ids

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/free_spaces')
def free_spaces():
    # Get the IDs of free parking spaces
    free_spaces_ids = [space_id for space_id, status in space_status.items() if status == 'free']
    # Return the IDs as JSON data
    return jsonify({'free_spaces': free_spaces_ids})

if __name__ == '__main__':
    app.run( debug=True)
