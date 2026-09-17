export function shouldSubmitOnEnter(event) {
  return event.key === 'Enter' && !event.shiftKey
    && !event.nativeEvent?.isComposing && !event.isComposing
    && (event.nativeEvent?.keyCode ?? event.keyCode) !== 229;
}
