from detect_activity_file import detect_activity
from detect_background_file import detect_back
from detect_obj_file import detect_obj
from gen_sentence_file import text_model

def final_summary(video_path):
    b=detect_back(video_path)
    o=detect_obj(video_path)
    a=detect_activity(video_path)
    out=[""]
    out1=[""]
    c_1=0
    c_2=0
    for i,j,k in zip(a,o,b):
        k=[z.split("/")[0] for z in k]
        obj=i[0].split()[0].lower()
        count=j.count(obj)
        count=3 if count>3 else count
        words=[i[0],k,f"{count} seconds"]
        sent=text_model(words,do_sample=False,num_beams=4)
        if sent == out[0]:
            c_1+=count
            out1[-1]=sent.replace("."," ")+f"for {c_1} seconds."
        else:
            c_1=count
            out[0]=sent
            out1.append(sent.replace("."," ")+f"for {c_1} seconds.")
    return out1[1:]