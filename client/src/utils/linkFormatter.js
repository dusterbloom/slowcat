/**
 * Client-side markdown link converter for stubborn LLMs
 * Converts [text](url) markdown links to clickable HTML
 */

/**
 * Convert markdown links to HTML in chat messages
 * @param {string} text - Text containing markdown links
 * @returns {string} - Text with HTML links
 */
export function convertMarkdownLinks(text) {
  if (!text || typeof text !== 'string') {
    return text;
  }
  
  // Regex to match [text](url) patterns
  const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  
  // Convert to clickable HTML links
  return text.replace(markdownLinkRegex, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

/**
 * Process chat message for display - handles both markdown and pre-formatted HTML
 * @param {string} message - Raw message from LLM
 * @returns {string} - Processed message with clickable links
 */
export function processMessageForDisplay(message) {
  if (!message) return message;
  
  // If already contains HTML links, return as-is
  if (message.includes('<a href=')) {
    return message;
  }
  
  // Convert markdown links to HTML
  return convertMarkdownLinks(message);
}

/**
 * Setup automatic link conversion for voice UI components
 * Call this once when the app initializes
 */
export function setupLinkConversion() {
  // Observer to watch for new chat messages
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          // Find chat message elements (adjust selector as needed)
          const messageElements = node.querySelectorAll('.message-content, [data-message], .chat-message');
          
          messageElements.forEach((element) => {
            const originalText = element.textContent;
            if (originalText && originalText.includes('[') && originalText.includes('](')) {
              const processedHTML = processMessageForDisplay(originalText);
              if (processedHTML !== originalText) {
                element.innerHTML = processedHTML;
              }
            }
          });
        }
      });
    });
  });
  
  // Start observing
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
  
  console.log('🔗 Link conversion observer active - markdown links will auto-convert to HTML');
  
  return observer;
}