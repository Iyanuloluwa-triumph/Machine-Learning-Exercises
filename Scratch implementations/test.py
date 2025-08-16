def text(string):
    result = ""
    for word in string.split(" "):
        if word[0].isalpha():
            word = word[1:] + word[0] + "ay"
            result += word + " "
        else:
            for item in word:
                if item.isalpha():
                    word = word.replace(item, " ") + item + "ay"
                    result += word + " "
                else:
                    result += word + " "
                    break
    return result.strip()


print(text("'Anointing fall on me"))
