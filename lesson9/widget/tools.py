def get_status_message(bmi: float) -> str:
    '''
    #param bmi: 這是要傳入的 bmi 值
    #return: 傳出 bmi 狀態
    '''

    if bmi < 18.5:
        return "體重過輕"
    elif bmi < 24:
        return "正常範圍"
    elif bmi < 27:
        return "過重"
    elif bmi < 30:
        return "輕度肥胖"
    elif bmi < 35:
        return "中度肥胖"
    else:
        return "重度肥胖"
    
class MyClass():
    @classmethod # @ 代表註冊一個class method
    def get_status_message(cls, bmi: float) -> str: #第一個引數一訂要加 cls
        '''
        #param bmi: 這是要傳入的 bmi 值
        #return: 傳出 bmi 狀態
        '''

        if bmi < 18.5:
            return "體重過輕"
        elif bmi < 24:
            return "正常範圍"
        elif bmi < 27:
            return "過重"
        elif bmi < 30:
            return "輕度肥胖"
        elif bmi < 35:
            return "中度肥胖"
        else:
            return "重度肥胖"