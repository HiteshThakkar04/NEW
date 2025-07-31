# def user_session(request):
#     print("session load ....................")
#     return {
#         'user_session': request.session.get('user_session', None)
#     }
def user_session(request):
    print("session load ....................")
    return {
        'user_session': request.session.get('user_session', None)
    }