// Extract only the static tool definition literals; do not run browser tools.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const root = process.argv[2];
const tools = ['calculator', 'dateTime', 'notes', 'codeRunner', 'search', 'webpage'].map(name => {
  const source = fs.readFileSync(path.join(root, 'src', 'tools', name + '.ts'), 'utf8');
  const match = source.match(/definition:\s*([\s\S]*?),\s*execute:/);
  if (!match) throw new Error('Missing definition: ' + name);
  return vm.runInNewContext('(' + match[1] + ')', Object.create(null), {timeout: 1000});
});
fs.writeFileSync('website-tools.json', JSON.stringify(tools, null, 2) + '\n');
