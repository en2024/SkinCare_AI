"""
Chatbot API — Gemini-powered skincare assistant.

Endpoints:
  POST /api/chat/       → Send a message, get an AI response
  POST /api/chat/clear/ → Clear the session chat history
"""
import os
import json
import logging
import traceback

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

SYSTEM_PROMPT = """
You are a professional skincare consultant for the AI Skincare Safety Analyzer app.
Your goal is to provide intelligent skincare advice, routines, ingredient analysis, and help with conditions like acne, eczema, and rosacea.

# Core Behavior
- Act like a professional skincare consultant.
- Answer naturally and avoid repetitive or robotic responses.
- Provide practical advice and explain ingredients simply.
- Recommend routines based on the user's skin type.
- NEVER claim to provide a medical diagnosis. Clearly state that you are an AI assistant and recommend consulting a dermatologist for serious concerns.
- Keep responses concise but helpful (under 300 words unless the user asks for details).
- Support both English and Arabic.
"""


@csrf_exempt
@require_POST
def chat_api(request):
    """POST /api/chat/ — Send a message to the Gemini chatbot."""
    if genai is None:
        return JsonResponse({
            'error': 'Google GenAI SDK is not installed. Please install it (pip install google-genai) to use the chatbot.'
        }, status=500)

    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
    except Exception:
        return JsonResponse({'error': 'Invalid request.'}, status=400)

    if not user_message:
        return JsonResponse({'error': 'Message cannot be empty.'}, status=400)

    # Load API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        env_path = os.path.join(settings.BASE_DIR, '.env')
        logger.error('Gemini API Key missing. Checked .env at: ' + str(env_path))
        return JsonResponse({
            'error': 'Gemini API Key is missing. Please configure it in the .env file located at ' + str(env_path) + '.'
        }, status=500)

    try:
        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        )

        # Retrieve session history
        history = request.session.get('chat_history', [])

        # Build formatted history for Gemini
        formatted_history = []
        for msg in history:
            formatted_history.append(
                types.Content(
                    role=msg['role'],
                    parts=[types.Part.from_text(text=msg['text'])]
                )
            )

        # Create chat and send message
        chat_session = client.chats.create(
            model='gemini-2.5-flash',
            config=config,
            history=formatted_history,
        )

        response = chat_session.send_message(user_message)
        ai_response = response.text

        # Save to session history
        history.append({'role': 'user', 'text': user_message})
        history.append({'role': 'model', 'text': ai_response})
        request.session['chat_history'] = history

        return JsonResponse({'response': ai_response})

    except Exception as e:
        logger.error('Chatbot API Error: ' + str(e))
        traceback.print_exc()
        return JsonResponse({
            'error': 'AI service temporarily unavailable or connection timed out. Please try again later.'
        }, status=500)


@csrf_exempt
@require_POST
def clear_chat(request):
    """POST /api/chat/clear/ — Clear the chat session history."""
    request.session['chat_history'] = []
    return JsonResponse({'status': 'cleared'})
