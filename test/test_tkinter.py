#calc.py
#coding=utf-8

from tkinter import *

#이벤트 처리함수
def enter(btn):
    if btn == 'C':
        ent.delete(0, END)
    elif btn == '=':
        ans = eval(ent.get())
        ent.delete(0, END)
        ent.insert(0, ans)
    else:
        ent.insert(END, btn)


def quit():
    root.destroy()
    root.quit()

#창만들기
root=Tk()
root.title('셈틀')
#root.protocol("WM_DELETE_WINDOW"), quit)

#숫자 입력란 만들기
ent=Entry(root)
ent.insert(0, ' ')
ent.pack(pady=5)

#숫자 버튼 만들기
buttons = ['7410', '852=', '963+', 'C/*-']
for col in buttons:
    frm=Frame(root)
    frm.pack(side=LEFT)
    for row in col :
        btn=Button(frm, text=row, command=(lambda char=row: enter(char)))
        btn.pack(fill=X, padx=5, pady=5)

#프로그램 실행
root.mainloop()