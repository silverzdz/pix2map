import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
from apscheduler.schedulers.blocking import BlockingScheduler

def email_msg(mail_sender, mail_receivers, subject, text=None, img_pth=None, att_pth=None):
    from_post = "{}<{}>".format(mail_sender.split('@')[0], mail_sender)
    to_post = []
    for post in mail_receivers:
        to_post.append("{}<{}>".format(post.split('@')[0], post))
    to_post = ','.join(to_post)

    mm = MIMEMultipart('related')
    mm["From"] = from_post
    mm["To"] = to_post
    # 邮件主题
    mm["Subject"] = Header(subject, 'utf-8')
    # 正文
    if text:
        message_text = MIMEText(text, "plain", "utf-8")
        mm.attach(message_text)
    # 添加图片
    if img_pth:
        image_data = open(img_pth, 'rb')
        message_image = MIMEImage(image_data.read())
        image_data.close()
        mm.attach(message_image)
    # 添加附件
    if att_pth:
        atta = MIMEText(open(att_pth, 'rb').read(), 'base64', 'utf-8')
        atta["Content-Disposition"] = 'attachment; filename="{}"'.format(os.path.basename(att_pth))
        mm.attach(atta)
    return mm

def send_email(mail_host, mail_sender, mail_receivers, port, mail_license, subject, text=None, img_pth=None, att_pth=None):
    stp = smtplib.SMTP_SSL(mail_host, port)
    #stp.connect(mail_host, port)
    #stp.set_debuglevel(1)
    stp.login(mail_sender, mail_license)
    msg = email_msg(mail_sender, mail_receivers, subject, text, img_pth, att_pth)
    stp.sendmail(mail_sender, mail_receivers, msg.as_string())
    print("邮件发送成功")
    stp.quit()

if __name__ == "__main__":
    kwargs = {
        "mail_host" : "smtp.qq.com",           # 发件邮箱smtp服务地址。此处用的是qq邮箱
        "mail_sender" : "1023814324@qq.com",        # 发件人邮箱
        "mail_receivers" : ["1023814324@qq.com"] ,  # 收件人邮箱
        "mail_license" : 'pssswmnidgnfbaji',                   # 邮箱授权码
        "port": 465,
        "subject" : "train_log",                 # 主题
        "text" : "train_log" ,                  # 正文
        "img_pth" : ''    ,                    # 图片路径
        "att_pth" : 'log.txt'  ,                      # 附件路径
    }
    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(send_email, 'interval', minutes = 60, start_date='2023-04-27 11:44:00', end_date='2023-05-03 23:59:59', kwargs=kwargs)

    try:
        scheduler.start()
        print("statistic scheduler start success")
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("statistic scheduler start-up fail")