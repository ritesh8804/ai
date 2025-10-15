// Background service worker

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'saveProfile') {
    saveProfileData(request.data).then(() => {
      sendResponse({ success: true });
    }).catch((error) => {
      console.error('Error saving profile:', error);
      sendResponse({ success: false, error: error.message });
    });
    return true; // Keep the message channel open for async response
  }

  if (request.action === 'getAllProfiles') {
    getAllProfiles().then((profiles) => {
      sendResponse({ success: true, profiles: profiles });
    }).catch((error) => {
      sendResponse({ success: false, error: error.message });
    });
    return true;
  }

  if (request.action === 'deleteProfile') {
    deleteProfile(request.profileId).then(() => {
      sendResponse({ success: true });
    }).catch((error) => {
      sendResponse({ success: false, error: error.message });
    });
    return true;
  }

  if (request.action === 'clearAllProfiles') {
    clearAllProfiles().then(() => {
      sendResponse({ success: true });
    }).catch((error) => {
      sendResponse({ success: false, error: error.message });
    });
    return true;
  }

  if (request.action === 'fetchUserData') {
    fetchUserDataFromSheet(request.linkedinUrl).then((data) => {
      sendResponse({ success: true, data: data });
    }).catch((error) => {
      sendResponse({ success: false, error: error.message });
    });
    return true;
  }
});

async function saveProfileData(profileData) {
  return new Promise((resolve, reject) => {
    chrome.storage.local.get(['profiles'], (result) => {
      const profiles = result.profiles || {};

      // Use profileId as key
      if (profileData.profileId) {
        profiles[profileData.profileId] = profileData;

        chrome.storage.local.set({ profiles: profiles }, () => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            console.log('Profile saved:', profileData.profileId);
            resolve();
          }
        });
      } else {
        reject(new Error('No profile ID found'));
      }
    });
  });
}

async function getAllProfiles() {
  return new Promise((resolve, reject) => {
    chrome.storage.local.get(['profiles'], (result) => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError);
      } else {
        resolve(result.profiles || {});
      }
    });
  });
}

async function deleteProfile(profileId) {
  return new Promise((resolve, reject) => {
    chrome.storage.local.get(['profiles'], (result) => {
      const profiles = result.profiles || {};

      if (profiles[profileId]) {
        delete profiles[profileId];

        chrome.storage.local.set({ profiles: profiles }, () => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve();
          }
        });
      } else {
        reject(new Error('Profile not found'));
      }
    });
  });
}

async function clearAllProfiles() {
  return new Promise((resolve, reject) => {
    chrome.storage.local.set({ profiles: {} }, () => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError);
      } else {
        resolve();
      }
    });
  });
}

// Fetch user data from Google Sheet
async function fetchUserDataFromSheet(linkedinUrl) {
  try {
    // Replace with your Google Apps Script Web App URL
    const SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzhYOl_Hpa7lub2dTZN1lM2SltUXLr0kMYeRT8VAUPgW_6PAffMjcKpxQ85cMaVSHxW/exec';
    
    const response = await fetch(`${SCRIPT_URL}?action=getbyurl&url=${encodeURIComponent(linkedinUrl)}`);
    const result = await response.json();
    
    if (result.success) {
      return result.data;
    } else {
      throw new Error(result.error || 'Failed to fetch data');
    }
  } catch (error) {
    console.error('Error fetching from Google Sheet:', error);
    // Return default structure on error
    return {
      name: '',
      job_title: '',
      company_name: '',
      phone: '',
      email: '',
      updated_at: new Date().toLocaleDateString(),
      metrics: [
        { source: 'Zobble', bounce: '0', unsubscribed: 'No', clicked: '0', enquired: '0' },
        { source: 'Violet', bounce: '0', unsubscribed: 'No', clicked: '0', enquired: '0' },
        { source: 'WOM', bounce: '0', unsubscribed: 'No', clicked: '0', enquired: '0' }
      ]
    };
  }
}
