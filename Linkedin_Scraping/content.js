// Content script that runs on LinkedIn profile pages

function waitForElement(selector, timeout = 5000) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(selector)) {
      return resolve(document.querySelector(selector));
    }

    const observer = new MutationObserver(() => {
      if (document.querySelector(selector)) {
        observer.disconnect();
        resolve(document.querySelector(selector));
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    setTimeout(() => {
      observer.disconnect();
      reject(new Error(`Timeout waiting for ${selector}`));
    }, timeout);
  });
}

function scrapeProfileData() {
  const data = {
    timestamp: new Date().toISOString(),
    url: window.location.href,
    profileId: window.location.pathname.split('/in/')[1]?.split('/')[0] || null
  };

  // Name - try multiple selectors
  try {
    const nameSelectors = [
      'h1.text-heading-xlarge',
      '.pv-text-details__left-panel h1',
      'h1',
      '[class*="top-card"] h1'
    ];

    for (const selector of nameSelectors) {
      const element = document.querySelector(selector);
      if (element && element.innerText.trim()) {
        data.name = element.innerText.trim();
        break;
      }
    }

    if (!data.name) data.name = null;
  } catch (e) {
    data.name = null;
  }

  // Headline - try multiple selectors
  try {
    const headlineSelectors = [
      '.text-body-medium.break-words',
      '.pv-text-details__left-panel .text-body-medium',
      '[class*="top-card"] [class*="headline"]',
      '.pv-top-card-profile-picture__container + div .text-body-medium'
    ];

    for (const selector of headlineSelectors) {
      const element = document.querySelector(selector);
      if (element && element.innerText.trim() && element.innerText !== data.name) {
        data.headline = element.innerText.trim();
        break;
      }
    }

    if (!data.headline) data.headline = null;
  } catch (e) {
    data.headline = null;
  }

  // Location - try multiple selectors
  try {
    const locationSelectors = [
      '.text-body-small.inline.t-black--light.break-words',
      '.pv-text-details__left-panel .text-body-small',
      '[class*="top-card"] .text-body-small',
      '.pv-top-card--list-bullet li:first-child'
    ];

    for (const selector of locationSelectors) {
      const element = document.querySelector(selector);
      if (element && element.innerText.trim()) {
        data.location = element.innerText.trim();
        break;
      }
    }

    if (!data.location) data.location = null;
  } catch (e) {
    data.location = null;
  }

  // About section
  try {
    const aboutHeader = Array.from(document.querySelectorAll('h2, div[id="about"]')).find(el =>
      el.textContent.toLowerCase().includes('about') || el.id === 'about'
    );

    if (aboutHeader) {
      const aboutSection = aboutHeader.closest('section') || aboutHeader.parentElement?.parentElement;

      // Try multiple selectors for the about text
      data.about = aboutSection?.querySelector('.inline-show-more-text')?.innerText?.trim() ||
                   aboutSection?.querySelector('.pv-shared-text-with-see-more')?.innerText?.trim() ||
                   aboutSection?.querySelector('.full-width span[aria-hidden="true"]')?.innerText?.trim() ||
                   aboutSection?.querySelector('div[class*="display-flex"] span')?.innerText?.trim() || null;
    } else {
      data.about = null;
    }
  } catch (e) {
    console.error('Error scraping about:', e);
    data.about = null;
  }

  // Profile picture URL
  try {
    const profileImg = document.querySelector('.pv-top-card-profile-picture__image, img[class*="profile-photo"]');
    data.profilePicture = profileImg?.src || null;
  } catch (e) {
    console.error('Error scraping profile picture:', e);
    data.profilePicture = null;
  }

  // Connections count
  try {
    const connectionsElement = Array.from(document.querySelectorAll('span, li')).find(el =>
      el.textContent.match(/\d+\+?\s*(connection|follower)/i)
    );
    data.connections = connectionsElement?.textContent?.trim() || null;
  } catch (e) {
    console.error('Error scraping connections:', e);
    data.connections = null;
  }

  // Current position (first experience item)
  try {
    if (data.experience && data.experience.length > 0) {
      const current = data.experience[0];
      if (current.duration && (current.duration.toLowerCase().includes('present') ||
          current.duration.toLowerCase().includes('current'))) {
        data.currentPosition = current.title;
        data.currentCompany = current.company;
      }
    }
  } catch (e) {
    console.error('Error extracting current position:', e);
  }

  // Experience - Be very specific to only get the Experience section
  try {
    data.experience = [];

    // Find the Experience section more precisely
    const experienceHeader = Array.from(document.querySelectorAll('h2, div[id="experience"]')).find(el =>
      el.textContent.toLowerCase().includes('experience') || el.id === 'experience'
    );

    if (experienceHeader) {
      // Get the parent container that has the list items
      const experienceSection = experienceHeader.closest('section') ||
                                 experienceHeader.parentElement?.parentElement;

      // Try different list selectors
      let experienceList = experienceSection?.querySelector('ul.pvs-list');

      if (!experienceList) {
        experienceList = experienceSection?.querySelector('ul');
      }

      if (!experienceList) {
        experienceList = experienceSection?.querySelector('div.pvs-list__outer-container');
      }

      let experienceItems = experienceList?.querySelectorAll(':scope > li') || [];

      // Try a broader selector if the specific one doesn't work
      if (experienceItems.length === 0) {
        experienceItems = experienceSection?.querySelectorAll('li') || [];
      }

      experienceItems.forEach(item => {
        // Get all visible text spans (filter out empty ones)
        const allSpans = Array.from(item.querySelectorAll('span[aria-hidden="true"]'))
          .map(s => s.innerText?.trim())
          .filter(text => text && text.length > 0);

        // Initialize fields
        let title = null;
        let company = null;
        let duration = null;
        let location = null;
        let employmentType = null;

        // Strategy: Look for bold elements for title/company
        const boldElement = item.querySelector('.mr1.t-bold, [class*="entity-result__title"]');
        const mainText = boldElement?.innerText?.trim() || allSpans[0];

        // Pattern recognition for LinkedIn's various layouts
        let spanIndex = 0;

        for (let i = 0; i < allSpans.length; i++) {
          const text = allSpans[i];

          // Skip if it's the main text we already captured
          if (text === mainText && i === 0) {
            title = mainText;
            continue;
          }

          // Check if it contains employment type indicators (Full-time, Part-time, Contract, etc.)
          if (text.match(/Full-time|Part-time|Contract|Freelance|Internship|Self-employed/i)) {
            // This span likely contains "Company · Type" or just "Type"
            const parts = text.split('·').map(p => p.trim());
            if (parts.length > 1) {
              company = parts[0];
              employmentType = parts[1];
            } else {
              employmentType = text;
            }
            continue;
          }

          // Check if it's a duration (contains months, years, or date patterns)
          if (text.match(/\d+\s*(mo|mos|month|months|yr|yrs|year|years)|[A-Z][a-z]{2}\s+\d{4}|\d{4}\s*-/i)) {
            if (!duration) {
              duration = text;
            }
            continue;
          }

          // Check if it's a location (contains comma, country names, or location keywords)
          if (text.includes(',') ||
              text.match(/on-site|remote|hybrid|algeria|france|usa|uk|canada/i)) {
            if (!location) {
              location = text;
            }
            continue;
          }

          // If we don't have company yet and this looks like a company name
          if (!company && !text.match(/\d/) && text.length > 2 && text.length < 100) {
            company = text;
          }
        }

        // Fallback: If no title found, first span is title
        if (!title && allSpans[0]) {
          title = allSpans[0];
        }

        // Get description if available
        const description = item.querySelector('.inline-show-more-text, .pvs-list__outer-container .visually-hidden')?.innerText?.trim() || null;

        const exp = {
          title: title,
          company: company,
          employmentType: employmentType,
          duration: duration,
          location: location,
          description: description
        };

        // Only add if we have at least title or company
        if (exp.title || exp.company) {
          data.experience.push(exp);
        }
      });

    }
  } catch (e) {
    console.error('Error scraping experience:', e);
    data.experience = [];
  }

  // Education - Be very specific to only get the Education section
  try {
    data.education = undefined;

    // Find the Education section more precisely
    const educationHeader = Array.from(document.querySelectorAll('h2, div[id="education"]')).find(el =>
      el.textContent.toLowerCase().includes('education') || el.id === 'education'
    );

    if (false) {
      const educationSection = educationHeader.closest('section') ||
                               educationHeader.parentElement?.parentElement;

      let educationList = educationSection?.querySelector('ul.pvs-list');
      if (!educationList) {
        educationList = educationSection?.querySelector('ul');
      }

      const educationItems = educationList?.querySelectorAll(':scope > li') || [];

      educationItems.forEach(item => {
        const allSpans = Array.from(item.querySelectorAll('span[aria-hidden="true"]'));

        const edu = {
          school: allSpans[0]?.innerText?.trim() || null,
          degree: allSpans[1]?.innerText?.trim() || null,
          duration: allSpans[2]?.innerText?.trim() || null
        };

        if (edu.school) {
          data.education.push(edu);
        }
      });

    }
  } catch (e) {
    console.error('Error scraping education:', e);
    data.education = undefined;
  }

  // Skills - Be very specific to only get the Skills section
  try {
    data.skills = undefined;

    // Find the Skills section more precisely
    const skillsHeader = Array.from(document.querySelectorAll('h2, div[id="skills"]')).find(el =>
      el.textContent.toLowerCase().includes('skills') || el.id === 'skills'
    );

    if (false) {
      const skillsSection = skillsHeader.closest('section') ||
                            skillsHeader.parentElement?.parentElement;

      let skillsList = skillsSection?.querySelector('ul.pvs-list');
      if (!skillsList) {
        skillsList = skillsSection?.querySelector('ul');
      }

      // Get all list items
      const skillItems = skillsList?.querySelectorAll(':scope > li') || [];

      skillItems.forEach(item => {
        // Look for the skill name - it's usually in a bold span or first visible span
        const boldSpan = item.querySelector('.mr1.t-bold span[aria-hidden="true"], [class*="entity-result__title"] span[aria-hidden="true"]');
        let skillName = boldSpan?.innerText?.trim();

        // If not found, get the first visible span
        if (!skillName) {
          const firstSpan = item.querySelector('span[aria-hidden="true"]');
          skillName = firstSpan?.innerText?.trim();
        }

        // Add skill with endorsement count if available
        if (skillName) {
          // Look for endorsement count
          const endorsementText = Array.from(item.querySelectorAll('span[aria-hidden="true"]'))
            .map(s => s.innerText?.trim())
            .find(text => text && text.match(/\d+\s*endorsement/i));

          const skillData = {
            name: skillName,
            endorsements: endorsementText || null
          };

          data.skills.push(skillData);
        }
      });

      // If we got very few skills, try alternate approach - sometimes skills are shown collapsed
      if (data.skills.length < 3) {
        // Try to find "Show all X skills" button and extract count
        const showAllButton = skillsSection?.querySelector('a[href*="/details/skills"]');
        if (showAllButton) {
          // Get the visible skills only
          const visibleSkills = Array.from(skillsSection.querySelectorAll('span[aria-hidden="true"]'))
            .map(s => s.innerText?.trim())
            .filter(text => text && text.length > 2 && text.length < 100 && !text.match(/endorsement|show all/i));

          // Add unique skills
          visibleSkills.forEach(skillName => {
            if (!data.skills.find(s => s.name === skillName)) {
              data.skills.push({ name: skillName, endorsements: null });
            }
          });
        }
      }

    }
  } catch (e) {
    console.error('Error scraping skills:', e);
    data.skills = undefined;
  }

  return data;
}

// Click all "Show all" / "Show more" buttons to expand sections
async function expandAllSections() {
  console.log('Expanding all sections...');

  // Look for "Show more" buttons within each section (inline expansion)
  const showMoreButtons = Array.from(document.querySelectorAll('button, a')).filter(btn => {
    const text = btn.textContent.toLowerCase().trim();
    const ariaLabel = btn.getAttribute('aria-label')?.toLowerCase() || '';

    return (text.includes('show all') ||
            text.includes('show more') ||
            text.includes('see more') ||
            text.includes('voir plus') ||
            text.match(/^\d+\s*more/) ||
            ariaLabel.includes('show all') ||
            ariaLabel.includes('show more')) &&
           !text.includes('post') && // Exclude "show more posts"
           !text.includes('activity'); // Exclude activity section
  });

  console.log(`Found ${showMoreButtons.length} expand buttons`);

  // Click each button with a delay
  let clickedCount = 0;
  for (const button of showMoreButtons) {
    try {
      // Check if button is still in DOM and visible
      if (document.body.contains(button) && button.offsetParent !== null) {
        button.scrollIntoView({ behavior: 'smooth', block: 'center' });
        await new Promise(resolve => setTimeout(resolve, 300));
        button.click();
        clickedCount++;
        console.log(`Clicked expand button: ${button.textContent.substring(0, 50)}`);
        await new Promise(resolve => setTimeout(resolve, 800)); // Wait for content to load
      }
    } catch (e) {
      console.error('Error clicking expand button:', e);
    }
  }

  console.log(`Clicked ${clickedCount} expand buttons`);

  // Wait for all content to finish loading
  await new Promise(resolve => setTimeout(resolve, 2000));
}

// Auto-scroll to load all content
async function autoScroll() {
  return new Promise((resolve) => {
    let totalHeight = 0;
    const distance = 300;
    const timer = setInterval(() => {
      const scrollHeight = document.body.scrollHeight;
      window.scrollBy(0, distance);
      totalHeight += distance;

      if (totalHeight >= scrollHeight) {
        clearInterval(timer);
        // Scroll back to top
        window.scrollTo(0, 0);
        resolve();
      }
    }, 100);
  });
}

// Auto-scrape when page loads
async function autoScrape() {
  try {
    // Wait for the page to load - try multiple selectors
    const possibleSelectors = [
      'h1.text-heading-xlarge',
      'h1',
      '.pv-text-details__left-panel h1',
      '[class*="top-card"]'
    ];

    let found = false;
    for (const selector of possibleSelectors) {
      try {
        await waitForElement(selector, 3000);
        found = true;
        break;
      } catch (e) {
        continue;
      }
    }

    if (!found) return;

    // First, expand all sections by clicking "Show all" buttons
    await expandAllSections();

    // Then auto-scroll to load any remaining lazy-loaded content
    await autoScroll();

    // Wait for content to render after scrolling
    await new Promise(resolve => setTimeout(resolve, 2000));

    const profileData = scrapeProfileData();

    // Send to background script for storage
    chrome.runtime.sendMessage({
      action: 'saveProfile',
      data: profileData
    }, (response) => {
      if (response?.success) {
        console.log('✓ LinkedIn profile scraped and saved successfully');
      }
    });
  } catch (error) {
    console.error('Error scraping profile:', error);
  }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'scrapeProfile') {
    const data = scrapeProfileData();
    sendResponse({ success: true, data: data });
  }
  return true;
});

// Auto-scrape when the page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', autoScrape);
} else {
  autoScrape();
}
