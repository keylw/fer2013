import src.haar-cascade


def test_face_detect():
    img = cv2.imread('hk_1.jpg')

    face, region = Face_detecter().get_face(img)
    
    