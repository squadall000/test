"""Real-time unevenness detection from a camera feed.

This script connects to the default system camera, processes each frame to
highlight surface irregularities, and displays the result. The background is set
to black and detected edges are painted bright red.
"""

import cv2


def main() -> None:
    """Capture video from the camera and highlight surface unevenness."""

    # Initialize video capture (0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        # Convert the frame to grayscale to simplify processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce image noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use the Canny operator to detect edges / surface unevenness
        edges = cv2.Canny(blur, 50, 150)

        # Prepare a result image: black background and red edges
        result = frame.copy()
        result[edges == 0] = [0, 0, 0]
        result[edges != 0] = [0, 0, 255]

        # Display the processed frame
        cv2.imshow("Detected Unevenness", result)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources when done
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
