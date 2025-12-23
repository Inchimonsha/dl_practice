from langchain_core.tools import tool


@tool
def schedule_agent():
    """
    Получает расписание мероприятий на указанную дату.
    Если дата не указана, возвращает расписание на текущую неделю.
    """
    print("schedule_agent")
    return {"answer": ['Monday', 'Tuesday']}


@tool
def news_agent():
    """
    Получает последние новости университета.
    """
    print("news_agent")
    return {"answer": "Hello world!"}
