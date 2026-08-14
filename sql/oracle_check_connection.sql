-- Diagnostic rapide de la connexion SQL Developer.
-- Executer avec F5.

show con_name;
show user;

select name, open_mode
from v$pdbs
order by name;

select sys_context('USERENV', 'CON_NAME') as current_container,
       sys_context('USERENV', 'SESSION_USER') as session_user
from dual;
