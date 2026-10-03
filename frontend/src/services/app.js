'use strict';
const $ = id => document.getElementById(id);
let metadata;
const colors = {'Low':'#226958','Medium':'#94783b','High':'#ab6443','Very High':'#964b56'};
const format = value => Number(value).toLocaleString(undefined,{maximumFractionDigits:2});
const percent = value => value == null ? 'N/A' : `${(value * 100).toFixed(1)}%`;
const cell = value => { const el = document.createElement('td'); el.textContent = value; return el; };
function appendRow(parent, values) { const tr = document.createElement('tr'); values.forEach(value => tr.append(cell(value))); parent.append(tr); }
async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(new URL(path, window.location.href), {...options, signal:controller.signal});
    const data = await response.json();
    if (!response.ok) {
      const message = Array.isArray(data.detail) ? data.detail.map(item => `${item.loc.slice(1).join('.')}: ${item.msg}`).join('; ') : data.detail;
      throw new Error(message || 'The assessment could not be completed.');
    }
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The request timed out. Check that the MindTrack server is still running.');
    throw error;
  } finally { clearTimeout(timer); }
}
function showError(message) { $('error').textContent = message; $('error').hidden = false; }
function bar(parent, label, value, caption, color='#669884') {
  const row = document.createElement('div'); row.className='bar-row';
  const header=document.createElement('div'); header.className='bar-label';
  const name=document.createElement('span'); name.textContent=label;
  const amount=document.createElement('strong'); amount.textContent=caption;
  header.append(name,amount);
  const track=document.createElement('div'); track.className='bar-track';
  const fill=document.createElement('div'); fill.className='bar-fill'; fill.style.width=`${Math.max(0,Math.min(100,value*100))}%`; fill.style.background=color;
  track.append(fill); row.append(header,track); parent.append(row);
}
function buildForm() {
  const inputTotal=metadata.numeric_features.length+metadata.categorical_features.length;
  $('input-count').textContent=`${inputTotal} inputs`;
  const accepted=metadata.bounds.Age; const observed=metadata.training_ranges.Age;
  $('cohort-note').textContent=`Accepted ages ${accepted[0]}–${accepted[1]} · training ages ${format(observed[0])}–${format(observed[1])}`;
  metadata.numeric_features.forEach(name => {
    const wrapper=document.createElement('div'); wrapper.className='field';
    const label=document.createElement('label'); label.htmlFor=name; label.textContent=metadata.display_names[name];
    const input=document.createElement('input'); input.id=name; input.name=name; input.type='number'; input.required=true;
    [input.min,input.max]=metadata.bounds[name]; input.step=['Age','Daily_Unlocks'].includes(name)?'1':'0.1';
    const help=document.createElement('small'); help.id=`${name}-help`;
    help.textContent=name==='Age'?`Accepted ${metadata.bounds.Age[0]}–${metadata.bounds.Age[1]}; training data ${format(metadata.training_ranges.Age[0])}–${format(metadata.training_ranges.Age[1])}`:['Study_Hours','Physical_Activity_Hours'].includes(name)?'Hours as recorded; source period is undocumented':'Typical recorded value';
    input.setAttribute('aria-describedby',help.id); wrapper.append(label,input,help); $('numeric-fields').append(wrapper);
  });
  metadata.categorical_features.forEach(name => {
    const wrapper=document.createElement('div'); wrapper.className='field';
    const label=document.createElement('label'); label.htmlFor=name; label.textContent=metadata.display_names[name];
    const select=document.createElement('select'); select.id=name; select.name=name; select.required=true;
    metadata.categories[name].forEach(choice=>{ const option=document.createElement('option'); option.value=choice; option.textContent=choice; select.append(option); });
    const other=document.createElement('option'); other.value=name==='Most_Used_Platform'?'Other':'Other / Not applicable'; other.textContent=other.value; select.append(other);
    wrapper.append(label,select); $('category-fields').append(wrapper);
  });
  resetForm(); $('fields').disabled=false;
}
function resetForm() {
  metadata.numeric_features.forEach(name=>{ $(name).value=['Age','Daily_Unlocks'].includes(name)?Math.round(metadata.medians[name]):Number(metadata.medians[name]).toFixed(1); });
  metadata.categorical_features.forEach(name=>{ $(name).selectedIndex=0; });
  if (metadata.categories.Academic_Level.includes('Undergraduate')) $('Academic_Level').value='Undergraduate';
  $('error').hidden=true; $('result').hidden=true; $('details').hidden=true;
  $('recommendations').hidden=true; $('empty-result').hidden=false;
}

/**
 * Render a single action card into the given parent element.
 * @param {HTMLElement} parent
 * @param {{category:string, priority:string, title:string, reason:string, action:string}} item
 */
function buildActionCard(parent, item) {
  const card = document.createElement('article');
  card.className = 'rec-card';
  card.setAttribute('aria-label', `${item.priority} priority: ${item.title}`);

  // Priority badge
  const badge = document.createElement('span');
  badge.className = `rec-badge ${item.priority}`;
  const dot = document.createElement('span'); dot.className = 'rec-badge-dot'; dot.setAttribute('aria-hidden', 'true');
  badge.append(dot, `${item.priority.toUpperCase()} PRIORITY`);

  // Category label
  const category = document.createElement('p');
  category.className = 'rec-category';
  category.textContent = item.category;

  // Title
  const title = document.createElement('h3');
  title.className = 'rec-title';
  title.textContent = item.title;

  // Plain-language reason
  const reasonLabel = document.createElement('p');
  reasonLabel.className = 'rec-action-label';
  reasonLabel.textContent = 'WHY THIS WAS SUGGESTED';
  const desc = document.createElement('p');
  desc.className = 'rec-desc';
  desc.textContent = item.reason;

  // Action box
  const actionBox = document.createElement('div');
  actionBox.className = 'rec-action';
  const actionLabel = document.createElement('p');
  actionLabel.className = 'rec-action-label';
  actionLabel.textContent = 'TRY THIS';
  const actionText = document.createElement('p');
  actionText.style.margin = '0';
  actionText.textContent = item.action;
  actionBox.append(actionLabel, actionText);

  card.append(badge, category, title, reasonLabel, desc, actionBox);
  parent.append(card);
}

/**
 * Render the full personalised recommendations section.
 * @param {{predicted_risk:string, threshold_source:string, summary:string, items:Array, disclaimer:string}} recs
 */
function displayRecommendations(recs) {
  if (!recs || !Array.isArray(recs.items)) return;

  $('rec-summary').textContent = recs.summary || '';
  $('rec-disclaimer').textContent = recs.disclaimer || '';

  // Build action cards
  const grid = $('rec-actions');
  grid.replaceChildren();
  recs.items.forEach(item => buildActionCard(grid, item));

  $('recommendations').hidden = false;
}

function renderResult(data) {
  $('empty-result').hidden=true; $('result').hidden=false; $('details').hidden=false;
  $('risk-label').textContent=data.label; $('risk-label').style.color=colors[data.label];
  $('result').style.borderTopColor=colors[data.label];
  $('primary-caption').textContent=`Estimated by ${data.primary_model}, selected using validation macro F1.`;
  $('probabilities').replaceChildren(); metadata.labels.forEach(label=>bar($('probabilities'),label,data.probabilities[label],percent(data.probabilities[label]),colors[label]));
  $('cluster-title').textContent=`Lifestyle cluster ${data.cluster.id + 1}`;
  $('cluster-description').textContent=data.cluster.description;
  $('model-predictions').replaceChildren(); data.models.forEach(item=>{ const row=document.createElement('div'); row.className='model-row'; const name=document.createElement('span'); name.textContent=item.model; const value=document.createElement('strong'); value.textContent=item.label; row.append(name,value); $('model-predictions').append(row); });
  $('agreement').textContent=`${data.agreement} of ${data.models.length} models agree with the primary estimate. Agreement does not establish correctness.`;
  $('comparison-rows').replaceChildren(); data.comparisons.forEach(item=>appendRow($('comparison-rows'),[item.name,format(item.value),format(item.training_median)]));
  $('result-notes').textContent=data.notes.join(' '); $('result-notes').hidden=!data.notes.length;
  // Render personalised recommendations if present
  if (data.recommendations) {
    displayRecommendations(data.recommendations);
    $('recommendations').scrollIntoView({behavior:'smooth',block:'start'});
  } else {
    $('result').scrollIntoView({behavior:'smooth',block:'nearest'});
  }
}
function renderResearch(metrics) {
  const split=metadata.split_sizes; $('split-info').textContent=`${split.train.toLocaleString()} training / ${split.val.toLocaleString()} validation / ${split.test.toLocaleString()} test records · Seed ${metadata.seed} · Primary model: ${metadata.primary_model}`;
  Object.entries(metrics).forEach(([name,item])=>appendRow($('metrics-rows'),[name,percent(item.accuracy),percent(item.precision_macro),percent(item.recall_macro),percent(item.f1_macro),item.roc_auc_ovr_macro==null?'N/A':item.roc_auc_ovr_macro.toFixed(3)]));
  const maximum=Math.max(...metadata.global_importance.map(item=>Math.abs(item.importance)),.001);
  metadata.global_importance.forEach(item=>bar($('importance-bars'),metadata.display_names[item.feature],Math.abs(item.importance)/maximum,item.importance.toFixed(3),item.importance<0?'#b89f85':'#669884'));
  metadata.cluster_scores.forEach(item=>appendRow($('cluster-rows'),[`${item.k}${item.k===metadata.cluster_count?' · selected':''}`,item.silhouette.toFixed(3),format(item.inertia)]));
  metadata.limitations.forEach(text=>{const li=document.createElement('li');li.textContent=text;$('limitations').append(li);});
}
document.querySelectorAll('.tab').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(tab=>{const active=tab===button;tab.classList.toggle('active',active);tab.setAttribute('aria-pressed',String(active));});
  $('assessment-panel').hidden=button.dataset.tab!=='assessment'; $('research-panel').hidden=button.dataset.tab!=='research';
}));
$('reset').addEventListener('click',resetForm);
$('assessment-form').addEventListener('submit',async event=>{
  event.preventDefault(); $('error').hidden=true;
  const payload={}; metadata.numeric_features.forEach(name=>payload[name]=Number($(name).value)); metadata.categorical_features.forEach(name=>payload[name]=$(name).value);
  $('fields').disabled=true; $('analyze').textContent='Analyzing your routine…';
  try {renderResult(await api('predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}));}
  catch(error){showError(error.message); $('result').hidden=true; $('details').hidden=true; $('recommendations').hidden=true; $('empty-result').hidden=false;}
  finally{$('fields').disabled=false;$('analyze').textContent='Explore my patterns ↗';}
});
(async()=>{try{const [info,metrics]=await Promise.all([api('model-info'),api('metrics')]);metadata=info;buildForm();renderResearch(metrics);$('loading').hidden=true;}catch(error){$('loading').hidden=true;showError(error.message+' Train or export the model, then restart the server or API cell.');}})();
