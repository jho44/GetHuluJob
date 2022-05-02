const puppeteer = require('puppeteer');
const fs = require('fs');
let movieTitles = [];
fs.readFile('./movieTitle.txt', 'utf8', async (err, data) => {
  if (err) {
    console.error(err);
    return;
  }
  movieTitles = data.split('\n');
  for (const movieTitle of movieTitles) {
    const script = await getScript(movieTitle);
    fs.appendFile('scripts.txt', script + '\n', function (err) {
      if (err) throw err;
      console.log(`Saved ${movieTitle}!`);
    });
  }
})

async function getScript(movieTitle) {
  const browser = await puppeteer.launch({headless: true});
  const page = await browser.newPage();

  const titleAndYr = movieTitle.split(/( \()|\)/g);
  const titleSansYr = titleAndYr[0];

  await page.goto('https://imsdb.com/');
  await page.type('input[name="search_query"]', movieTitle);
  await page.click('input[type="submit"]');
  try {
    await page.waitForSelector('p > a', {timeout: 5000});
  } catch (error) {
    await browser.close();
    return "None";
  }
  const firstResultTitle = await page.$$eval('p > a', async anchor => {
    return anchor[0].text;
  });

  if (!firstResultTitle.includes(titleSansYr)) {
    await browser.close();
    return "None";
  }
  await page.click('p > a'); // click on the first result
  try {
    await page.waitForSelector('.script-details', {timeout: 5000});
  } catch (error) {
    await browser.close();
    return "None";
  }
  await page.$$eval('.script-details', table => {
    table[0].lastElementChild.lastElementChild.lastElementChild.lastElementChild.click();
  });
  try {
    await page.waitForSelector('.scrtext > pre', {timeout: 5000});
  } catch (error) {
    await browser.close();
    return "None";
  }
  const script = await page.$$eval('.scrtext > pre', pre => {
    return pre[0].textContent.replace(/(<([^>]+)>)/g, "").replace(/(\s\s+)|\n/g, ' ');
  })
  await browser.close();

  return script;
}