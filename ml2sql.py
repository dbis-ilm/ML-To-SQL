import psycopg2
import pyodbc
from tensorflow import keras

class ML2SQL:
    """ Converting a Tensorflow ML model into a relational model table and allow model inference 
        using pure SQL. Dense layers and LSTM layers are allowed.
        The logical model is mapped to a physical model that is stored in the 
        backend DBMS. The physical model might have different layers (e.g. an additional input layer, 
        two physical layers for a logical LSTM layer).
        Each node is identified using a unique node_id.

    Attributes:
        odbc_con                ODBC connection object
        backend                 String indicating 'vector' or 'postgres'
        model                   Tensorflow model
        unique_node_mapping     Enumeration of all nodes, organized as array of arrays
    """
    odbc_con = None
    backend = None
    model = None
    unique_node_mapping = None


    def __run_update_query(self, query):
        cursor = self.odbc_con.cursor()
        cursor.execute(query)
        self.odbc_con.commit()
        cursor.close()

    def _unique_node_mapping(self, model):
        mapping = []
        current = 0

        # input layer
        neurons = model.layers[0].get_config().get("units")
        layer_mapping = [0] * neurons
        for i in range(neurons):
            layer_mapping[i] = current
            current +=1
        mapping.append(layer_mapping)

        # other layers
        for layer in model.layers:
            runs = 1
            if isinstance(layer, keras.layers.LSTM):
                runs = 2 # we split LSTM layer into two

            neurons = layer.get_config().get("units")
            for i in range(runs):
                layer_mapping = [0] * neurons
                for i in range(neurons):
                    layer_mapping[i] = current
                    current +=1
                mapping.append(layer_mapping)
        return mapping

    def __init__(self, odbc_con, backend, model):
        assert backend in ("postgres", "vector")

        self.backend = backend
        self.odbc_con = odbc_con
        self.model = model
        self.unique_node_mapping = self._unique_node_mapping(model)
        queries = []

        # Create mathematical functions
        if backend == "postgres":
            queries.append("CREATE OR REPLACE FUNCTION sigmoid(float) RETURNS float\
                AS 'select 0.5 * (1 + tanh($1/2))'\
                LANGUAGE SQL")
        elif backend == "vector":
            queries.append("CREATE OR REPLACE FUNCTION cosh(x FLOAT8)RETURN(FLOAT8) AS\
                BEGIN RETURN (EXP(x)+EXP(-x))/2;END")
            queries.append("CREATE OR REPLACE FUNCTION sinh(x FLOAT8)RETURN(FLOAT8) AS BEGIN RETURN (EXP(x)-EXP(-x))/2;END")
            queries.append("CREATE OR REPLACE FUNCTION tanh(x FLOAT8)RETURN(FLOAT8) AS BEGIN RETURN sinh(x)/cosh(x); END")
            queries.append("CREATE OR REPLACE FUNCTION sigmoid(x FLOAT8) RETURN (FLOAT8) AS\
                BEGIN RETURN 0.5 * (1 + tanh(x/2)); END")
        else:
            assert False
        for q in queries:
            self.__run_update_query(q)

    def __get_column_data_type(self, table_name, column_name):
        cursor = self.odbc_con.cursor()

        if self.backend == "postgres":
            q = f"select data_type from information_schema.columns \
            where table_name='{table_name}' and column_name='{column_name}'\
            order by table_schema, table_name;"
            cursor = self.odbc_con.cursor()
            cursor.execute(q)
            rs = cursor.fetchone()
            col_type = rs[0]
        elif self.backend == "vector":
            # This should be the portable way, but psycopg2 does not support it
            column_info = cursor.columns(table = table_name, column=column_name).fetchone()
            assert column_info, "Could not get column type"
            col_type = column_info.type_name
            
        else:
            assert False

        cursor.close()
        return col_type

    # Activation functions
    # Input table by query of schema (id, node_id, output) grouped by (id, node_id), output summed up
    # Output table by query of schema (id, node_id, output_activated)
    def __linear_activation(self, query):
        q = f"Select id, node, output as output_activated from ({query}) as t"
        return q

    def __sigmoid_activation(self, query):
        q = f"Select id, node, sigmoid(output) as output_activated from ({query}) as t"
        return q

    def __relu_activation(self, query):
        q = f"Select id, node, case when output > 0 then output else 0.0 end as output_activated from ({query}) as t"
        return q

    def __input_layer_to_relation(self, model_table_name, first_layer):
        values = []
        queries = []
        
        first_layer_units = first_layer.get_config().get("units")
        for n_idx in range(first_layer_units):
            node_id = self.unique_node_mapping[0][n_idx]
            values.append(f"(-1, {node_id}, 1, 0)")

        value_str = ','.join(values)   
        # Save weight and bias in W_i and b_i
        queries.append(f"INSERT INTO {model_table_name} (Node_IN, Node, W_i, b_i) \
            VALUES {value_str};")
        
        return queries
            
    def __dense_layer_to_relation(self, model_table_name, layer, physical_layer_idx):
        queries = []
        values = []
        kernel, bias = layer.weights
        kernel = kernel.numpy().tolist()
        bias = bias.numpy().tolist()

        for n_idx, neuron in enumerate(kernel):
            for e_idx, edge in enumerate(neuron):
                node_in_id = self.unique_node_mapping[physical_layer_idx][n_idx]
                node_id = self.unique_node_mapping[physical_layer_idx + 1][e_idx]
                values.append(f"({node_in_id}, {node_id}, {edge}, {bias[e_idx]})")
        
        value_str = ','.join(values)   
        # Save weight and bias in W_i and b_i
        queries.append(f"INSERT INTO {model_table_name} (Node_IN, Node, W_i, b_i) \
            VALUES {value_str};")
        
        return queries

    def __lstm_layer_to_relation(self, model_table_name, layer, physical_layer_idx):
        queries = []
        values = []
        
        # slice the weights to get the gate matrices
        units = int(int(layer.trainable_weights[0].shape[1])/4)

        W = layer.get_weights()[0]
        U = layer.get_weights()[1]
        b = layer.get_weights()[2]

        W_i = W[:, :units]
        W_f = W[:, units: units * 2]
        W_c = W[:, units * 2: units * 3]
        W_o = W[:, units * 3:]

        U_i = U[:, :units]
        U_f = U[:, units: units * 2]
        U_c = U[:, units * 2: units * 3]
        U_o = U[:, units * 3:]

        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]

        num_neurons = layer.get_config().get("units")

        # divide into two layers: 
        # First layer
        # TODO: explanation
        for i in range(num_neurons):
            node_in_id = self.unique_node_mapping[physical_layer_idx][i]
            node_id = self.unique_node_mapping[physical_layer_idx + 1][i]
            values.append(f"({node_in_id}, {node_id}, " + 
                        f"{W_i[0, i]}, {W_f[0, i]}, {W_c[0, i]},{W_o[0, i]}, " +
                        f"{b_i[i]}, {b_f[i]}, {b_c[i]}, {b_o[i]})")
        
        value_str = ','.join(values)   
        queries.append(f"INSERT INTO {model_table_name} (Node_IN, Node, " + 
                    "W_i, W_f, W_c, W_o, b_i, b_f, b_c, b_o) " +
                    f"VALUES {value_str};")
        
        # Second layer
        physical_layer_idx += 1
        values = []
        for i in range(num_neurons):
            for k in range(num_neurons):
                node_in_id = self.unique_node_mapping[physical_layer_idx][i]
                node_id = self.unique_node_mapping[physical_layer_idx + 1][k]
                values.append(f"({node_in_id}, {node_id}, " + 
                            f"{U_i[i, k]}, {U_f[i, k]}, {U_c[i, k]},{U_o[i, k]})")
        
        value_str = ','.join(values)   
        queries.append(f"INSERT INTO {model_table_name} (Node_IN, Node, " + 
                    "U_i, U_f, U_c, U_o) " +
                    f"VALUES {value_str};")    
        return queries


    # Model import query construction
    def model_to_relation(self, model_table_name):
        queries = []
        queries.append(f"Drop table if exists {model_table_name};")

        max_node_id = self.unique_node_mapping[-1][-1]

        if max_node_id < 32767:
            node_type = "SMALLINT"
        elif max_node_id < 2147483647:
            node_type = "INTEGER"
        else:
            node_type = "BIGINT"

        queries.append(f"CREATE TABLE {model_table_name} (\
            Node_IN {node_type} not null,\
            Node {node_type} null,\
            W_i float4,\
            W_f float4,\
            W_c float4,\
            W_o float4,\
            U_i float4,\
            U_f float4,\
            U_c float4,\
            U_o float4,\
            b_i float4,\
            b_f float4,\
            b_c float4,\
            b_o float4\
            );"
        )

        layers = self.model.layers
        values = []
        
        # Input layer
        queries.extend(self.__input_layer_to_relation(model_table_name, layers[0]))
        
        # Remaining layers
        physical_layer_idx = 0
        for layer in layers:
            if isinstance(layer, keras.layers.Dense):
                queries.extend(self.__dense_layer_to_relation(model_table_name, layer, physical_layer_idx))
                physical_layer_idx += 1
            elif isinstance(layer, keras.layers.LSTM):
                queries.extend(self.__lstm_layer_to_relation(model_table_name, layer, physical_layer_idx))
                physical_layer_idx += 2 # LSTM maps to 2 layers
            else:
                assert False, "Layer type currently not supported"
        
        return queries

    def prepare_time_series_table(self, table_name, id_col_name, val_col_name, time_steps, ts_table_name):
        val_col_type = self.__get_column_data_type(table_name, val_col_name)
        val_cols = [val_col_name]

        q = f"Drop table if exists {ts_table_name}"
        self.__run_update_query(q)

        q = f"Create table {ts_table_name} as (Select {id_col_name}, {val_col_name} from {table_name})"
        self.__run_update_query(q)

        if self.backend == "postges":
            q = f"Alter table {ts_table_name} add primary key ({id_col_name})"
            self.__run_update_query(q)
        

        add_cols_queries = []
        update_queries = []

        postgres_from_clause = ""
        vector_from_clause = ""
        if self.backend == "postgres":
            postgres_from_clause = f"from {ts_table_name} as tab"
        elif self.backend == "vector":
            vector_from_clause = f"from {ts_table_name} as tab"
        else:
            assert False

        for i in range(1, time_steps):
            add_cols_queries.append(f"Alter table {ts_table_name} add column {val_col_name}_{i} {val_col_type}")
            update_queries.append(f"Update {ts_table_name} {vector_from_clause} set {val_col_name}_{i} = tab.{val_col_name} \
                {postgres_from_clause} where {ts_table_name}.{id_col_name} = tab.{id_col_name} + {i}")
            val_cols.append(f"{val_col_name}_{i}")

        for q in add_cols_queries: 
            self.__run_update_query(q)
        for q in update_queries: 
            self.__run_update_query(q)
            
        return val_cols

    # Modeljoin query construction of schema (id, node_id, [outputs]) (but no activation applied)
    # returns query and list of output names
    def __input_join_query(self, input_query, id_col, input_cols, model_table_name):
        col_assignments = []
        case_statements = []
        output_names = []
        
        if isinstance(input_cols[0], list):
            assert len(input_cols) == 1, "If input cols is a list, only one element is allowed"
            output = ", ".join(input_cols[0])
            col_assignment_str = output
            output_names = input_cols[0]
        else:
            for idx, col in enumerate(input_cols):
                col_assignments.append(f"{col} as C{idx}")
                case_statements.append(f"when node={idx} then C{idx}")
        
            col_assignment_str = ",".join(col_assignments)
            case_statement_str = "\n".join(case_statements)
            output_names = ["output_activated"]
            output = f"case\
                {case_statement_str}\
                end as output_activated"
            
        # Cross join between data and model - one inference per input tuple
        cross_j = f"select data.{id_col} as id, {col_assignment_str}, node\
            from ({input_query}) as data, {model_table_name} as model where model.node_in = -1"
        
        input_q = f"Select id, node, \
            {output} from ({cross_j}) as t"
        
        return (input_q, output_names)

    # Input table by query of schema (id, node_id, output_activated)
    # Output table by query of schema (id, node_id, output) grouped by (id, node_id), output summed up
    def __dense_layer_query(self, query, model_table_name):
        # Trick: group by bias, its constant per node anyway, but we need it in outer query
        layer_q = f" Select id, node, s + bias as output from(\
            Select id, model.node as node, sum(input.output_activated * model.W_i) as s, model.b_i as bias\
            from ({query}) as input, {model_table_name} as model\
            where input.node = model.node_in\
            group by id, model.node, model.b_i) as t"
        
        return layer_q

    def __dense_layer_activation(self, query, layer):
        act = layer.get_config().get('activation')
        if act == 'linear':
            q = self.__linear_activation(query)
        elif act == 'relu':
            q = self.__relu_activation(query)
        elif act == 'sigmoid':
            q = self.__sigmoid_activation(query)
        else:
            assert False, "Activation currently not supported"
        return q


    # Input table by query of schema (id, node_id, [outputs])
    # Output table by query of schema (id, node_id, output) grouped by (id, node_id)
    def __lstm_layer_query(self, query, model_table_name, col_names, physical_layer_idx):
        time_steps = len(col_names)
        layer_units = len(self.unique_node_mapping[physical_layer_idx])

        # +1 offset because unique_node_mapping holds an artificial input layer
        this_layer_min_node_id = self.unique_node_mapping[physical_layer_idx][0]
        this_layer_max_node_id = self.unique_node_mapping[physical_layer_idx][-1]
        recurrent_layer_min_node_id = self.unique_node_mapping[physical_layer_idx + 1][0]
        recurrent_layer_max_node_id = self.unique_node_mapping[physical_layer_idx + 1][-1]

        value = col_names[-1:][0]
        remaining_cols = col_names[:-1]
        remaining_cols_str = ", ".join(remaining_cols)
        if remaining_cols:
            remaining_cols_str = ", " + ", ".join(remaining_cols)
        else:
            remaining_cols_str = ""

        # In the first time step, we do not need to consider the cell state
        # Each time step produces a new intermediate result of schema (id, layer, node, h, c)
        q = f"Select id, node, o * tanh(c) as h, c {remaining_cols_str} from (\
                Select id, node, i * tanh(z_c) as c, o  {remaining_cols_str} from (\
                    Select id, node, sigmoid(z_i) as i, sigmoid(z_o) as o, z_c  {remaining_cols_str} from (\
                        Select id, model.node as node, {value} * W_i + b_i as z_i, {value} * W_c + b_c as z_c,\
                        {value} * W_o + b_o as z_o {remaining_cols_str} from ({query}) as input, {model_table_name} as model\
                        where input.node = model.node_in and model.node >= {this_layer_min_node_id} and model.node <= {this_layer_max_node_id}\
                    ) as t\
                ) as t\
            ) as t"

        # Each time step produces a new intermediate result of schema (id, layer, node, h, c)
        for i in range(1, time_steps):
            # Compute the inner states
            # Having a filter for the reccurent layer node_id somehow slows down, but for the cell state
            # query it leads to acceleration. 
            # TODO: Why?
            # and model.node >= {recurrent_layer_min_node_id} and model.node <= {recurrent_layer_max_node_id}
            q = f"Select id, node, Sum(h * U_i) as state_i, Sum(h * U_f) as state_f, \
                Sum(h * U_c) as state_c, Sum(h * U_o) as state_o, sum(c) as c {remaining_cols_str} from (\
                    Select id, model.node as node, h, \
                    case when model.node_in = model.node - {layer_units} then c else 0 end as c, U_i, U_f, U_c, U_o {remaining_cols_str}\
                    from ({q}) as input, {model_table_name} as model\
                    where input.node = model.node_in\
                ) as t group by id, node {remaining_cols_str}"

            value = remaining_cols[-1:][0]
            remaining_cols = remaining_cols[:-1]
            if remaining_cols:
                remaining_cols_str = ", " + ", ".join(remaining_cols)
            else:
                remaining_cols_str = ""
            
            # Compute the new cell states
            # layer - 2 for recurrence
            q = f"Select id, node, o * tanh(c) as h, c {remaining_cols_str} from (\
                    Select id, node, f * c + i * tanh(z_c) as c, o  {remaining_cols_str} from (\
                        Select id, node, sigmoid(z_i) as i, sigmoid(z_f) as f, sigmoid(z_o) as o, z_c {remaining_cols_str}, c from (\
                            Select id, model.node as node, {value} * W_i + state_i + b_i as z_i, \
                            {value} * W_f + state_f + b_f as z_f, {value} * W_c + state_c + b_c as z_c,\
                            {value} * W_o + state_o + b_o as z_o {remaining_cols_str}, c from ({q}) as input, {model_table_name} as model\
                            where input.node - {2 * layer_units} = model.node_in and model.node >= {this_layer_min_node_id} and model.node <= {this_layer_max_node_id}\
                        ) as t\
                    ) as t\
                ) as t"
        
        # Last, we do not need to compute the inner states, but simply pass the result to the next layer
        q = f"Select id, node + {layer_units} as node, h as output_activated from ({q}) as t"

        return q

    # Nest queries, starting from applying the input to the model + a model forward and an activation query per layer
    # Join the result back with the table
    # TODO: Multiple output columns, different aggregation functions (not only sum up inputs)
    def model_join_query(self, input_query, id_col, input_cols, model_table_name, output_col_name):
        (q, col_names) = self.__input_join_query(input_query, id_col, input_cols, model_table_name)

        physical_layer_idx = 1 # Start at 1 after input layer
        for layer in self.model.layers:
            # Layer forward query
            if isinstance(layer, keras.layers.Dense):
                q = self.__dense_layer_query(q, model_table_name)
                q = self.__dense_layer_activation(q, layer)
                physical_layer_idx += 1
            elif isinstance(layer, keras.layers.LSTM):
                q = self.__lstm_layer_query(q, model_table_name, col_names, physical_layer_idx)
                physical_layer_idx += 2
            else:
                assert False, "Layer type currently not supported"
                
        # Join back with table
        q = f"Select data.id, data.*, input.output_activated as {output_col_name}\
        from ({q}) as input, ({input_query}) as data\
        where input.id = data.{id_col}"
        
        return q