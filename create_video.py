from shutil import copyfile
import cv2
import random



# load doc into memory
def read_document(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def create_video():
    writer = None

    filename = "Dataset/general_captions/Flickr8k.token.txt"
    text = read_document(filename)
    filenames = list()
    for line in text.split("\n"):
        if "cycle" in line:
            if line.split(" ")[0].split(".")[0] not in filenames:
                filenames.append(line.split(" ")[0].split(".")[0])
    count = 0
    # for file in filenames:
    #     src = "Dataset/general_images/"+file+".jpg"
    #     dest = "Cycle_images/cycle_"+str(count)+".jpg"
    #     copyfile(src, dest)
    #     print(src, " ====> ", dest)
    #     count += 1

    filename = "Dataset/general_captions/Flickr_8k.trainImages.txt"
    text = read_document(filename)
    trainfilenames = list()

    for line in text.split("\n"):
            if line.strip() not in filenames:
                trainfilenames.append(line.strip())

    train_count = 1
    cycle_count = 0
    frame_count = 0

    for i in range(1000):
        if train_count%5 != 0:
            frame_name = "Dataset/general_images/"+trainfilenames[random.randint(0,2000)]
            type = "train"
            train_count += 1
        else:
            frame_name = "Cycle_Images/cycle_"+str(cycle_count)+".jpg"
            type = "cycle"
            cycle_count += 1
            train_count += 1
        frame = cv2.imread(frame_name)
        frame = cv2.resize(frame, (300, 300))
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            writer = cv2.VideoWriter("created_video.mp4", fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        for j in range(30):
            # write the output frame to disk
            writer.write(frame)
            frame_count += 1
            # cv2.imshow("window", frame)
        print(frame_count)

    writer.release()


create_video()