-- Creation des tables uniquement.
-- A executer connecte en tant que IDS_APP sur le service xepdb1.

set define off;

declare
    procedure drop_table_if_exists(p_table varchar2) is
    begin
        execute immediate 'drop table ' || p_table || ' cascade constraints purge';
    exception
        when others then
            if sqlcode != -942 then
                raise;
            end if;
    end;
begin
    drop_table_if_exists('IDS_PREDICTIONS');
    drop_table_if_exists('IDS_ALERTS');
    drop_table_if_exists('IDS_UPLOADS');
    drop_table_if_exists('IDS_MODEL_RESULTS');
end;
/

create table ids_uploads (
    upload_id varchar2(36) primary key,
    source_type varchar2(20) not null,
    filename varchar2(255),
    model_name varchar2(100) not null,
    sample_count number not null,
    attack_count number not null,
    benign_count number not null,
    alert_rate number(10, 6) not null,
    status varchar2(20) not null,
    metrics_json clob,
    created_at timestamp default systimestamp not null
);

create table ids_predictions (
    prediction_id number generated always as identity primary key,
    upload_id varchar2(36) not null,
    row_number number not null,
    predicted_class number(1) not null,
    predicted_label varchar2(20) not null,
    attack_probability number(12, 10),
    benign_probability number(12, 10),
    created_at timestamp default systimestamp not null,
    constraint fk_ids_predictions_upload_app
        foreign key (upload_id) references ids_uploads(upload_id)
        on delete cascade
);

create table ids_alerts (
    alert_id number generated always as identity primary key,
    upload_id varchar2(36) not null,
    severity varchar2(20) not null,
    message varchar2(500) not null,
    created_at timestamp default systimestamp not null,
    constraint fk_ids_alerts_upload_app
        foreign key (upload_id) references ids_uploads(upload_id)
        on delete cascade
);

create table ids_model_results (
    model_id varchar2(80) primary key,
    model_name varchar2(120) not null,
    model_type varchar2(40) not null,
    accuracy number(12, 10),
    precision_value number(12, 10),
    recall_value number(12, 10),
    f1_score number(12, 10),
    attack_f1_score number(12, 10),
    roc_auc number(12, 10),
    updated_at timestamp default systimestamp not null
);

create index idx_ids_uploads_created_at on ids_uploads(created_at);
create index idx_ids_predictions_upload on ids_predictions(upload_id);
create index idx_ids_alerts_created_at on ids_alerts(created_at);

insert into ids_model_results values ('decision_tree', 'Decision Tree', 'Classique', 0.9626061001, 0.9438264300, 0.9952413437, 0.9688522466, 0.9532264734, 0.9814586686, systimestamp);
insert into ids_model_results values ('random_forest', 'Random Forest', 'Classique', 0.9613324129, 0.9413096811, 0.9959235287, 0.9678467770, 0.9515076878, 0.9832736257, systimestamp);
insert into ids_model_results values ('linear_svm', 'Linear SVM', 'Classique', 0.9121641987, 0.8832305222, 0.9791351225, 0.9287134651, 0.8856079216, 0.9368778260, systimestamp);
insert into ids_model_results values ('cnn', 'CNN', 'Deep learning', 0.6134991395, 0.8429568207, 0.4160995657, 0.5571695927, 0.6571152552, 0.7747163398, systimestamp);
insert into ids_model_results values ('lstm', 'LSTM', 'Deep learning', 0.8576387201, 0.8553538765, 0.9103176320, 0.8819802682, 0.8206472476, 0.8918373068, systimestamp);
insert into ids_model_results values ('autoencoder', 'Autoencoder', 'Deep learning', 0.7150829841, 0.7253186228, 0.8247616512, 0.7718503293, 0.6207092933, 0.7464245783, systimestamp);
insert into ids_model_results values ('cnn_lstm', 'CNN + LSTM', 'Deep learning', 0.8840944668, 0.8761124122, 0.9336783082, 0.9039798310, 0.8538215350, 0.9067635731, systimestamp);
insert into ids_model_results values ('cnn_lstm_attention', 'CNN + LSTM + Attention', 'Deep learning', 0.5698243090, 0.8826311471, 0.3043044209, 0.4525747940, 0.6457078796, 0.6726169982, systimestamp);

commit;

select 'INSTALLATION_TABLES_IDS_APP_OK' as status from dual;
