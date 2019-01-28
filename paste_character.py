import numpy as np
import cv2
import os
import random

class PASTE_CHARACTER(object):
    def __init__(self):
        self.char_lis = None


    def load_character(self, CHARA_CTER_PATH):
        
        char_file = os.listdir(CHARA_CTER_PATH)
        
        char_list = [None] * len(char_file)
        k = 0
        for f in char_file:
            img_path = CHARA_CTER_PATH + f
            img = cv2.imread(img_path,0)

            x_min = 10000
            y_min = 10000

            x_max = 0
            y_max = 0

            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if (img[y][x] < 200):
                        x_min = min(x, x_min)
                        y_min = min(y, y_min)
                        x_max = max(x, x_max)
                        y_max = max(y, y_max)
            
            img = img[y_min:y_max,x_min:x_max]
            
            char_list[k] = img
            k = k + 1

        self.char_lis = char_list
        random.shuffle(self.char_lis)

        return


    def paste(self, BACK_GROUND_PATH, LABEL_PATH, SAVE_PATH):
        back_ground_file = os.listdir(BACK_GROUND_PATH)
        cnt = 0

        for f in back_ground_file:
            back_img_path = BACK_GROUND_PATH + f
            label_path = LABEL_PATH + f[:-4] + '.npy'
            save_path = SAVE_PATH + f

            back_img = cv2.imread(back_img_path)
            label = np.load(label_path)
            
            char_num = 0


            for k in range(label.shape[0]):
                poi_1 = (label[k][0], label[k][1])
                poi_2 = (label[k][2], label[k][3])
                poi_3 = (label[k][4], label[k][5])
                poi_4 = (label[k][6], label[k][7])


                char = self.char_lis[char_num]
                char_num = (char_num + 1) % len(self.char_lis)

                poi_1_char = (0,0)
                poi_2_char = (char.shape[1],0)
                poi_3_char = (char.shape[1], char.shape[0])
                poi_4_char = (0,char.shape[0])

                coor = np.zeros((3,4))
                coor[0] = np.concatenate((poi_1, poi_1_char))
                coor[1] = np.concatenate((poi_2, poi_2_char))
                coor[2] = np.concatenate((poi_4, poi_4_char))
                A_1,B_1 = self.get_mat_A_B(coor)

                coor[0] = np.concatenate((poi_3, poi_3_char))
                coor[1] = np.concatenate((poi_2, poi_2_char))
                coor[2] = np.concatenate((poi_4, poi_4_char))
                A_2,B_2 = self.get_mat_A_B(coor)

                poi_1_lis = np.array([[poi_1_char[0], poi_1_char[1]] , [poi_2_char[0], poi_2_char[1]] ,[poi_4_char[0], poi_4_char[1]] ]).astype(int)
                poi_2_lis = np.array([[poi_3_char[0], poi_3_char[1]] , [poi_2_char[0], poi_2_char[1]] ,[poi_4_char[0], poi_4_char[1]] ]).astype(int)

                hull_1 = cv2.convexHull(poi_1_lis)
                hull_2 = cv2.convexHull(poi_2_lis)


                color = random.randint(0, 100)
                color = np.array([color, color, color])
                for y in range(char.shape[0]):
                    for x in range(char.shape[1]):
                        if char[y][x] < 200:
                            #try:
                            
                            # back_img[yy][xx] = color
                            #except:
                            #    continue
                            if cv2.pointPolygonTest(hull_1, (x,y), False) >= 0:
                                char[y][x] = 0
                                now = np.mat([x, y]).T
                                correspond = np.dot(A_1, now) + B_1
                                xx = int(correspond[0])
                                yy = int(correspond[1])
                                back_img[yy][xx] = color
                            elif cv2.pointPolygonTest(hull_2, (x,y), False) >= 0:
                                char[y][x] = 255
                                now = np.mat([x, y]).T
                                correspond = np.dot(A_2, now) + B_2
                                xx = int(correspond[0])
                                yy = int(correspond[1])
                                back_img[yy][xx] = color
                            


            
            for k in range(label.shape[0]):
                    poi_1 = ( int(label[k][0]), int(label[k][1]) )
                    poi_2 = ( int(label[k][2]), int(label[k][3]) )
                    poi_3 = ( int(label[k][4]), int(label[k][5]) )
                    poi_4 = ( int(label[k][6]), int(label[k][7]) )

                    cv2.line(back_img, poi_1, poi_2, (0,0,255), 1 )
                    cv2.line(back_img, poi_2, poi_3, (0,0,255), 1 )
                    cv2.line(back_img, poi_3, poi_4, (0,0,255), 1 )
                    cv2.line(back_img, poi_4, poi_1, (0,0,255), 1 )
            cv2.imshow('new_img', back_img)
            cv2.waitKey(0)



    def get_mat_A_B(self, coor):
        
        x1_B = coor[0][0]
        y1_B = coor[0][1]
        x1_A = coor[0][2]
        y1_A = coor[0][3]

        x2_B = coor[1][0]
        y2_B = coor[1][1]
        x2_A = coor[1][2]
        y2_A = coor[1][3]

        x3_B = coor[2][0]
        y3_B = coor[2][1]
        x3_A = coor[2][2]
        y3_A = coor[2][3]

        mat_1 = [[x1_B - x2_B, x1_B - x3_B], [y1_B - y2_B, y1_B - y3_B]]
        mat_1 = np.mat(mat_1)

        mat_2 = [[x1_A - x2_A, x1_A - x3_A], [y1_A - y2_A, y1_A - y3_A]]
        mat_2 = np.mat(mat_2)

        mat_2 =np.linalg.inv(mat_2)

        A = np.dot(mat_1, mat_2)

        Y = [x1_B, y1_B]
        Y = np.mat(Y).T

        X = [x1_A, y1_A]
        X = np.mat(X).T

        B = Y - np.dot(A, X)
        return A, B






if __name__ == '__main__':
    BACK_GROUND_PATH = '/home/ffb/workspace/python-srf/SynTextData/background/'
    LABEL_PATH = '/home/ffb/workspace/python-srf/SynTextData/label/'
    SAVE_PATH = '/home/ffb/workspace/python-srf/SynTextData/img/'
    CHARA_CTER_PATH = '/home/ffb/workspace/python-srf/SynTextData/character/'

    paste_char = PASTE_CHARACTER()
    paste_char.load_character(CHARA_CTER_PATH)
    paste_char.paste(BACK_GROUND_PATH, LABEL_PATH, SAVE_PATH)